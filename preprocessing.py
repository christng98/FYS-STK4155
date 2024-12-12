import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import kagglehub
import numpy as np
import librosa as lb
import soundfile as sf

class Preprocessor:
    def __init__(self):
        self.file_path = None
        self.df = None
        pass

    def download_data(self, path):
        self.dataset_dir = kagglehub.dataset_download(path)
        self.file_path = os.path.join(self.dataset_dir, 'Respiratory_Sound_database/Respiratory_Sound_Database/audio_and_txt_files')
        print("Path to dataset files:", self.file_path)
    
    def load_data(self):
        def _get_filename_info(file):
            return file.split('_')
        
        all_files = os.listdir(self.file_path)
        self.txt_files = [file for file in all_files if file.endswith('.txt')]
        self.wav_files = [file for file in all_files if file.endswith('.wav')]

        dataframes = []

        for file in self.txt_files:
            file_path = os.path.join(self.file_path, file)
            filename_info = _get_filename_info(file)

            df = pl.read_csv(file_path, separator='\t', has_header=False, new_columns=['Start', 'End', 'Crackles', 'Wheezes'])
            
            df = df.with_columns([
                pl.lit(filename_info[0]).alias('Patient_ID'),
                pl.lit(filename_info[1]).alias('Recording_ID'),
                pl.lit(filename_info[2]).alias('Location'),
                pl.lit(filename_info[3]).alias('Equipment'),
                pl.lit(file).alias('File')
            ])

            dataframes.append(df)

        self.df = pl.concat(dataframes)

    def plot_cycle_lengths(self):
        df = self.df
        start_times = df.get_column('Start')
        end_times = df.get_column('End')
        cycle_lengths =  end_times - start_times
        cycle_lengths = cycle_lengths.to_numpy()
        plt.hist(cycle_lengths, bins=100)
        plt.xlabel('Cycle Length')
        plt.ylabel('Count')
        plt.title('Distribution of Cycle Lengths')
        plt.show()

    def plot_cycle_lengths_by_patient(self):
        df = self.df
        start_times = df.get_column('Start')
        end_times = df.get_column('End')
        cycle_lengths =  end_times - start_times
        cycle_lengths = cycle_lengths.to_numpy()
        patients = df.get_column('Patient_ID').to_numpy()
        unique_patients = df.get_column('Patient_ID').unique().to_numpy()
        sns.scatterplot(x=cycle_lengths, y=patients)
        plt.xlabel('Cycle Length')
        plt.ylabel('Patient ID')
        plt.yticks(rotation=45)
        plt.title('Distribution of Cycle Lengths by Patient')
        plt.show()

    def process_audio_files(self, max_len):
        def _get_pure_sample(audio_array, start, end, sample_rate):
            max_index = len(audio_array)
            start_index = min(int(start * sample_rate), max_index)
            end_index = min(int(end * sample_rate), max_index)
            return audio_array[start_index:end_index]
        
        output_dir = 'Project_3/dataset/processed_audio_files'
        os.makedirs(output_dir, exist_ok=True)
        
        i, c = 0,0
        for index, row in enumerate(self.df.iter_rows()):
            start, end, crackles, wheezes, patient_id, recording_id, location, equipment, filename = row
            if end - start > max_len:
                end = start + max_len
            audio_file = os.path.join(self.file_path, filename.replace('.txt', '.wav'))
            if index > 0:
                # check if more cycles exist for the same patient, if so add i to change filename
                if self.df[index-1, 'Patient_ID'] == row[4]:  # row[4] is patient_id
                    i += 1
                else:
                    i = 0
            new_filename = f"{patient_id}_{i}.wav"
            new_audio_file = os.path.join(output_dir, new_filename)
            c += 1
            audio_array, sample_rate = lb.load(audio_file)
            pure_sample = _get_pure_sample(audio_array, start, end, sample_rate)
            # pad the sample if it is shorter than max_len
            required_len = max_len * sample_rate
            padded_sample = lb.util.pad_center(pure_sample, size=required_len)
            
            # Check if the file already exists
            if os.path.exists(new_audio_file):
                continue
            else:
                sf.write(new_audio_file, padded_sample, sample_rate)

        print(f"Processed {c} audio files")

    def create_dataset(self):
        for i,f in enumerate(os.listdir(self.audio_filepath)):
            pass
        pass