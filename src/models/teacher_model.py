# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import sys
import traceback
import time
from datetime import datetime
from nixtla import NixtlaClient

# Add the 'src/utils' directory to the module search path
#sys.path.append('..')
sys.path.append('/data/jfialho/git/2024/distillation/nexus/src')

from utils.nexutil import *  # Module from 'src/utils' for utils functions
from utils.nexloss import *  # Module from 'src/utils' for losses functions
from utils.nexdata import *  # Module from 'src/utils' for data functions
from models.student_model import StudentModel

class TeacherModel():
    def __init__(self,
                config_path=None,
                models=None,
                to_root_dir='.'):
        
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        self.logger = logging.getLogger(__name__)

        if models is None: models = []
        self.models = models
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp para nomear os arquivos de cache
        self.flag_nixtla = None 

        # Load the configuration parameters from the YAML file
        self.config = load_yaml_config(config_path)
        self.config_path = config_path

        self.logger.info(f'Loading config. file {self.config_path}')
        
        self.data_params = self.config['data']
        self.model_params = self.config['model']

        # Set seeds for reproducibility
        set_random_seeds(self.data_params['default_seed'])
        self.logger.info(f'Default random seed: {self.data_params["default_seed"]}')

        # Define the device
        try:
            self.device = torch.device(self.model_params['device'])
            self.logger.info(f'Default device for forecasting: {self.model_params["device"]}')
        except Exception as e:
            # Error messsage and the complete stack trace
            print(f"An error occurred when we try to set torch.device following de config file: {e}")
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raise

        # Define paths
        self.processed_dir = os.path.join(to_root_dir, self.data_params['processed_path'])
        self.interim_dir = os.path.join(to_root_dir, self.data_params['intermediate_path'])
        self.logger.info('Defining paths...')
        
        # Load dataframes from source
        self.train_df, self.test_df = load_processed_data(self.processed_dir,
                                                self.data_params['processed_train_df'],
                                                self.data_params['processed_test_df'])
        self.logger.info('Loading dataframes from source...')
        self.logger.info(f'Shape train_df: {self.train_df.shape}')
        self.logger.info(f'Shape test_df:  {self.test_df.shape}')

        window_mode = "fixed"

        # Crop target datetime
        self.crop_datetime = self.data_params['crop_target_datetime']
        if window_mode == "sliding":
            self.target_index = self.train_df.loc[self.train_df.ds == self.crop_datetime, 'ds'].index.values[0] - self.model_params['context_window_len']
        if window_mode == "fixed":    
            self.target_index = self.model_params['context_window_len']

        self.logger.info(f'target_index: {self.target_index}')
        if (self.target_index - self.model_params['context_window_len']) < 0:
            self.logger.error('target_index - context_window_len must be greater than 0')
            sys.exit(1)
        self.logger.info('Croping target datetime...')
        self.logger.info(f'Shape train: {self.train_df.shape}')
        self.logger.info(f'self.target_index: {self.target_index}')
        self.logger.info(f'### crop_target_datetime :  {self.data_params["crop_target_datetime"]}')
        self.logger.info(f'### Proposal context size:  {self.train_df.shape[0] - self.train_df.loc[self.train_df.ds == self.crop_datetime, "ds"].index.values[0]}')

        # Processed series
        self.processed_series = np.concatenate([self.train_df[self.target_index:].y.values,
                                                self.test_df.y.values], axis=None)
        self.logger.info('Concatenating Processed series...')
        
        # Datestamp from Processed series
        self.processed_series_ds =  np.concatenate([self.train_df[self.target_index:].ds.values,
                                                    self.test_df.ds.values], axis=None)
        self.logger.info('Get Datestamp from Processed series...')

        # Create input and output sequences
        self.X, self.y = create_sequences(self.processed_series,
                                        self.model_params['context_window_len'],
                                        self.model_params['forecast_len'])
        self.logger.info('Creating input and output sequences...')
        self.logger.info(f'len(X): {len(self.X)}')
        self.logger.info(f'len(y): {len(self.y)}')

        ## Create input and output date sequences
        self.X_ds, self.y_ds = create_sequences(self.processed_series_ds,
                                                self.model_params['context_window_len'],
                                                self.model_params['forecast_len'])
        self.logger.info('Creating input and output date sequences...')
        print(f'Primeiro elemento X[0] : {self.X_ds[0][0]}')
        print(f'Ultimo elemento X[0]   : {self.X_ds[0][-1]}')
        print(f'Primeiro elemento y[0] : {self.y_ds[0][0]}')
        print(f'Ultimo elemento y[0]   : {self.y_ds[0][-1]}')
        print(f'Primeiro elemento X[-1] : {self.X_ds[-1][0]}')
        print(f'Ultimo elemento X[-1]   : {self.X_ds[-1][-1]}')
        print(f'Primeiro elemento y[-1] : {self.y_ds[-1][0]}')
        print(f'Ultimo elemento y[-1]   : {self.y_ds[-1][-1]}')

        real_pivot_aux = self.train_df.loc[self.train_df.ds == self.crop_datetime, 'ds'].index.values[0]
        real_pivot = real_pivot_aux - self.model_params['forecast_len'] - 1
        print(f'###################### real_pivot   : {real_pivot}')

        # Calculate the index to split data into training and test sets
        self.split_index = self.train_df.shape[0] - self.target_index -\
            (self.model_params['context_window_len'] + self.model_params['forecast_len'] - 1)
        self.logger.info('Calculating the index to split data into training and test sets...')

        # Split input and output sequences into training and test sets
        self.X_train, self.X_test = split_data(self.X, self.split_index)
        self.y_train, self.y_test = split_data(self.y, self.split_index)
        self.logger.info('Split input and output sequences into training and test sets..')

        #TODO: Fixing timegpt function to work with nexdata function
        if 'timegpt' in self.models:
            # SSH from Processed series
            self.processed_series_ssh =  np.concatenate([self.train_df[self.target_index:].ssh.values,
                                                        self.test_df.ssh.values], axis=None)
            self.logger.info('Get SSH from Processed series...')

            self.X_ssh, self.y_ssh = create_sequences(self.processed_series_ssh,
                                                    self.model_params['context_window_len'],
                                                    self.model_params['forecast_len'])
            self.logger.info('Creating input and output date sequences SSH...')

            # AT from Processed series
            self.processed_series_at =  np.concatenate([self.train_df.loc[self.target_index:,'at'].values,
                                                        self.test_df.loc[:,'at'].values], axis=None)
            self.logger.info('Get Astronomical Tide (at) from Processed series...')

            self.X_at, self.y_at = create_sequences(self.processed_series_at,
                                                    self.model_params['context_window_len'],
                                                    self.model_params['forecast_len'])
            self.logger.info('Creating input and output date sequences for Astronomical Tide...')

            try:
                nixtla_api_key = load_api_key(self.model_params['nixtla_api_key_path'])
                self.nixtla_client = NixtlaClient(api_key=nixtla_api_key)
                self.flag_nixtla = self.nixtla_client.validate_api_key()
                if self.flag_nixtla:
                    print('Nixtla TimeGPT-1 is ready to use :B')
                else:
                    print('Nixtla TimeGPT-1 is not available =/')
            except Exception as e:
                # Error messsage and the complete stack trace
                print(f"An error occurred when we try to initialize TimeGPT-1 API: {e}")
                traceback.print_exc()
                raise

        if 'chronos' in self.models:
            try:
                self.chronos_pipeline = ChronosPipeline.from_pretrained(
                self.model_params['chronos_t5_model'],
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                )
                self.logger.info(f'Chronos model are initialized with {self.model_params["chronos_t5_model"]} running on {self.model_params["device"]} device.')
            except Exception as e:
                # Error messsage and the complete stack trace
                print(f"An error occurred when we try to create an Chronos instance: {e}")
                traceback.print_exc()
                raise
        self.logger.info('End of init class method.')

    def save_results(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    #TODO: Fixing timegpt function to work with nexdata functions
    '''
    def timegpt(self, context_window_len=None, forecast_len=None,
                finetune_steps=0, endg=None, exog=None,
                cache_prefix="forecast_cache"):
        
        if self.flag_nixtla is not None:
            if exog is not None and endg is not None:
                fcst_y_df_all = []
                fcst_y_all = []
                fcst_y_df = []
                fcst_y = []
                num_calls = len(endg['y'])
                #num_calls = 10
                progress_bar = tqdm(total=num_calls, desc="Calls Progress")  # Barra de progresso
                
                for i in range(num_calls):
                    df_endg = pd.DataFrame()
                    list_endg = []

                    for endg_idx in endg:

                        if endg_idx == 'ds':
                            df_endg[endg_idx] = pd.to_datetime(endg[endg_idx][i])
                        else:
                            df_endg[endg_idx] = endg[endg_idx][i]

                        list_endg.append(endg_idx)

                    df_exog = pd.DataFrame()
                    list_exog = []

                    for exog_idx in exog:

                        if exog_idx == 'ds':
                            df_exog[exog_idx] = pd.to_datetime(exog[exog_idx][i])
                        else:
                            df_exog[exog_idx] = exog[exog_idx][i]

                        list_exog.append(exog_idx)
                    for _ in range(30): 
                        try:
                            print(f"{i} -> Try: {_} |  endog: {df_endg.loc[0,'ds']} | exog: {df_exog.loc[0,'ds']}")

                            fcst = self.nixtla_client.forecast(
                                df=df_endg,
                                X_df=df_exog,
                                h=forecast_len,
                                finetune_steps=0
                            )

                            fcst_y_df_all.append(fcst)
                            fcst_y_all.append(fcst.TimeGPT.values)
                            #print('fiz os 2 appends')
                            if (i + 1) % 10 == 0 or i == num_calls - 1:
                                #begin if
                                filename_fcst_df = f"{cache_prefix}_fcst_y_df_validated_{self.timestamp}.pkl"
                                filename_fcst = f"{cache_prefix}_fcst_y_validated_{self.timestamp}.pkl"
                                self.save_results(fcst_y_df_all, filename_fcst_df)
                                self.save_results(fcst_y_all, filename_fcst)
                                progress_bar.set_postfix({'Remaining Calls': num_calls - i - 1})
                                progress_bar.update(10)
                                #end if
                            break  # Exit the loop
                        except Exception as e:
                            # If an exception occurs, print an error message
                            # and continue the loop to try again
                            print(f"Error executing the block: {e}")
                            time.sleep(15)
                    else:
                        # If the loop completes 30 iterations without
                        # success, raise an error
                        raise RuntimeError("Failed to execute the\
                                           block after 30 attempts")

                progress_bar.close()
                return fcst_y_df_all, fcst_y_all
            else:
                print('Unavailable `endg` or `exog` input data')
        else:
            print('Nixtla TimeGPT-1 is not available =/')
    '''

    def timegpt(self):
        self.timegpt_predictions = []
        list_timegpt_ioa = []
        num_calls = len(self.X)

        self.timegpt_filename = f'{self.data_params["timegpt_cache_prefix"]}_{self.timestamp}.pkl'
        self.timegpt_file_path = os.path.join(self.interim_dir, self.timegpt_filename)

        self.logger.info(f'Creating file {self.timegpt_filename} ...')

        progress_bar = tqdm(total=num_calls, desc="Calls Progress")  # Barra de progresso
        
        for i in range(num_calls):
        # for i in range(11):
            for _ in range(self.model_params['attempts_after_failure']): 

                df_endg = pd.DataFrame()
                df_endg['ds'] = pd.to_datetime(self.X_ds[i])
                df_endg['y'] = self.X[i]
                df_endg['ssh'] = self.X_ssh[i]
                df_endg['at'] = self.X_at[i]


                df_exog = pd.DataFrame()
                df_exog['ds'] = pd.to_datetime(self.y_ds[i])
                df_exog['at'] = self.y_at[i]

                try:
                    fcst = self.nixtla_client.forecast(
                        df=df_endg,
                        X_df=df_exog,
                        h=self.model_params['forecast_len'],
                        finetune_steps=self.model_params['timegpt_finetune_steps']
                    )
                    if i==0:
                        print(df_endg)
                        print(df_exog)

                    predictions = np.array(fcst.TimeGPT.values)
                    self.timegpt_predictions.append(np.array(fcst.TimeGPT.values))
                    ioa = calculate_ioa(self.y[i],
                                        predictions[0])
                    list_timegpt_ioa.append(ioa)
                    ioa = round(ioa, 4) 
                    ioa_mov_avg = round(np.mean(list_timegpt_ioa), 4) 
                    print(f"{i} -> Try: {_} |  endog: {self.X_ds[i][0]} | IoA: {ioa} | IoA mov_AVG: {ioa_mov_avg} | ft steps: {self.model_params['timegpt_finetune_steps']}")
                    if (i + 1) % 10 == 0 or i == num_calls - 1:
                        #begin if
                        if i==0: self.logger.info(f'Saving results at {self.timegpt_file_path} ...')
                        try:
                            self.save_results(self.timegpt_predictions, self.timegpt_file_path)
                            if i == num_calls - 1 or (i + 1) % 100 == 0:
                                self.logger.info(f'TimeGPT-1 predict file are created with successfull at: {self.timegpt_file_path}')
                        except Exception as e_save:
                            self.logger.info(f'An error occured at saving step for file {self.timegpt_file_path}: {e_save}')        
                        progress_bar.set_postfix({'Remaining Calls': num_calls - i - 1})
                        progress_bar.update(10)
                        #end if
                    break  # Exit the loop
                except Exception as e:
                    # If an exception occurs, print an error message
                    # and continue the loop to try again
                    print(f"Error executing the block: {e}")
                    time.sleep(5)
            else:
                # If the loop completes 30 iterations without
                # success, raise an error
                raise RuntimeError("Failed to execute the\
                                    block after 30 attempts")
        progress_bar.close()
        return self.timegpt_predictions

    def chronos(self):
        self.chronos_predictions = []
        ioa_list = []
        num_calls = len(self.X)

        self.chonos_filename = f'{self.data_params["chronos_cache_prefix"]}_{self.timestamp}.pkl'
        self.chronos_file_path = os.path.join(self.interim_dir, self.chonos_filename)

        self.logger.info(f'Creating file {self.chonos_filename} ...')

        progress_bar = tqdm(total=num_calls, desc="Calls Progress")  # Barra de progresso
        
        for i in range(num_calls):
        # for i in range(55):
            for _ in range(self.model_params['attempts_after_failure']): 
                try:
                    batch_context = torch.tensor(self.X[i])
                    forecast = self.chronos_pipeline.predict(batch_context,self.model_params['forecast_len'])
                    predictions = np.quantile(forecast.numpy(), 0.5, axis=1)
                    self.chronos_predictions.append(np.array(predictions[0])) # check if works
                    ioa = calculate_ioa(self.y[i],
                                        predictions[0])
                    ioa_list.append(ioa)
                    ioa = round(ioa, 4) 
                    ioa_mov_avg = round(np.mean(ioa_list), 4) 
                    print(f"{i} -> Try: {_} |  endog: {self.X_ds[i][0]} | IoA: {ioa} | IoA mov_AVG: {ioa_mov_avg}")
                    if (i + 1) % 10 == 0 or i == num_calls - 1:
                        #begin if
                        if i==0: self.logger.info(f'Saving results at {self.chronos_file_path} ...')
                        try:
                            self.save_results(self.chronos_predictions, self.chronos_file_path)
                            if i == num_calls - 1 or (i + 1) % 100 == 0:
                                self.logger.info(f'Chronos predict file are created with successfull at: {self.chronos_file_path}')
                        except Exception as e_save:
                            self.logger.info(f'An error occured at saving step for file {self.chronos_file_path}: {e_save}')        
                        progress_bar.set_postfix({'Remaining Calls': num_calls - i - 1})
                        progress_bar.update(10)
                        #end if
                    break  # Exit the loop
                except Exception as e:
                    # If an exception occurs, print an error message
                    # and continue the loop to try again
                    print(f"Error executing the block: {e}")
                    time.sleep(5)
            else:
                # If the loop completes 30 iterations without
                # success, raise an error
                raise RuntimeError("Failed to execute the\
                                    block after 30 attempts")
        progress_bar.close()
        self.chronos_predictions = self.chronos_predictions
        return self.chronos_predictions

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('to_root_dir', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(config_path,
        to_root_dir,
        ):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    teacher_model = TeacherModel(models=['chronos'],
                                config_path=config_path,
                                to_root_dir=to_root_dir)
    #chronos_predictions = teacher_model.chronos()
    #timegpt_predictions = teacher_model.timegpt()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
