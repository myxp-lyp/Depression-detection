import os
import zipfile
import pandas as pd
from pydub import AudioSegment
import shutil

# 设置路径
download_dir = r'D:\Download\DAIC-WOZ'
audio_dir = os.path.join(download_dir, 'AUDIO')
preprocessed_audio_dir = os.path.join(download_dir, 'preprocessed_audio')

# 创建所需的文件夹
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(preprocessed_audio_dir, exist_ok=True)

# 解压压缩文件
def unzip_files():
    for i in range(300, 493):  # 遍历300到493（修改了范围）
        if i in [342, 394, 398, 460]:
            continue  # 跳过缺失的压缩包

        zip_file = os.path.join(download_dir, f"{i}_P.zip")
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(download_dir, str(i)))  # 解压到相应文件夹

        print(f"Participant {i} unzipped!")

# 处理每个参与者的数据
def process_audio():
    for i in range(300, 493):  # 遍历300到493（修改了范围）
        if i in [342, 394, 398, 460]:
            continue  # 跳过缺失的压缩包

        folder = os.path.join(download_dir, str(i))
        transcript_file = os.path.join(folder, f"{i}_TRANSCRIPT.csv")
        audio_file = os.path.join(folder, f"{i}_AUDIO.wav")

        if os.path.exists(transcript_file) and os.path.exists(audio_file):
            # 读取transcript csv文件，使用tab作为分隔符
            try:
                df = pd.read_csv(transcript_file, sep='\t')
                # 筛选出参与者（Participant）的行
                participant_data = df[df['speaker'] == 'Participant']
                audio = AudioSegment.from_wav(audio_file)

                n = 1
                # 遍历所有参与者的讲话段
                for index, row in participant_data.iterrows():
                    start_time = row['start_time'] * 1000  # 转换为毫秒
                    stop_time = row['stop_time'] * 1000  # 转换为毫秒

                    # 切割音频并保存
                    segment = audio[start_time:stop_time]
                    segment_filename = os.path.join(audio_dir, f"{i}_{n}.wav")
                    segment.export(segment_filename, format="wav")
                    n += 1
            except pd.errors.ParserError as e:
                print(f"Error reading {transcript_file}: {e}")
        
        print(f"Participant {i} processed!")

# 合成每个参与者的音频文件
def combine_audio():
    for i in range(300, 493):  # 遍历300到493（修改了范围）
        if i in [342, 394, 398, 460]:
            continue  # 跳过缺失的压缩包

        participant_audio_files = []
        # 获取该参与者的所有音频段文件
        for n in range(1, 1000):  # 假设最多有1000段
            segment_filename = os.path.join(audio_dir, f"{i}_{n}.wav")
            if os.path.exists(segment_filename):
                participant_audio_files.append(segment_filename)
            else:
                break

        # 将每5段音频文件合并成一段
        for j in range(0, len(participant_audio_files), 5):
            audio_segments = []
            for k in range(j, min(j + 5, len(participant_audio_files))):
                audio_segments.append(AudioSegment.from_wav(participant_audio_files[k]))

            # 合并音频段
            combined_audio = sum(audio_segments)
            combined_audio_filename = os.path.join(preprocessed_audio_dir, f"{i}_{(j//5) + 1}.wav")
            combined_audio.export(combined_audio_filename, format="wav")

            # 删除已合成的单段音频文件
            for file_path in participant_audio_files[j:j+5]:  # Delete the original files, not AudioSegments
                os.remove(file_path)
        
        print(f"Participant {i} processed!")

# 主函数执行
def main():
    unzip_files()
    print("解压完成！")
    process_audio()
    print("数据第一步处理完成！")
    combine_audio()
    print("数据预处理完成！")

if __name__ == "__main__":
    main()