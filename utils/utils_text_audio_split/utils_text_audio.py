from pydub import AudioSegment
import jsonlines
import os
import glob


def get_text_json_results(file):
    # 文本标注json 文件解析
    # 返回结果：[[text, label],[text, label],[text, label],...]
    results = []
    with open(file, 'r+', encoding='utf8') as f:
        for lines in jsonlines.Reader(f):
            for line in lines:
                # print('----原始数据-----')
                # print(line['data']['value'])
                for annotations in line['annotations']:
                    for result in annotations['result']:
                        # print(result['value']['start'])
                        # print(result['value']['end'])
                        # print(result['value']['text'])
                        # print(result['value']['labels'][0])
                        text = result['value']['text']
                        label = result['value']['labels'][0]
                        results.append([text, label])
    return results


def get_audio_result_json(file):
    data = []
    with open(file, 'r+', encoding='utf8') as f:
        for lines in jsonlines.Reader(f):
            for line in lines:
                print('----原始数据-----')
                # print(line['data']['audio'])
                for annotations in line['annotations']:
                    for result in annotations['result']:
                        # print(result)
                        # print(result['value']['start'])
                        # print(result['value']['end'])
                        start = int(result['value']['start'])
                        end = int(result['value']['end'])
                        if 'text' in result['value'].keys():
                            # print(result['value']['text'][0])
                            text = result['value']['text'][0]
                            data.append([start, end, text])
                        if 'labels' in result['value'].keys():
                            print(result['value']['labels'][0])
    return data


def split_sound(audio_file, json_file):
    """
    :param audio_file: mp3...其他格式文件，音频标注文件json
    :param json_file: 以标注内容为文件名，对音频文件进行切割，保存
    :return:
    """
    # TODO 不同文件格式文件的读取方式,未完待续...
    """
    audio_format = ['wav', 'mp3', 'mp4', 'wma', 'aac', 'ogg', 'flv']
    p = os.path.abspath(path)
    files = sorted(glob.glob(os.path.join(p, '*.*')))
    audios = [x for x in files if x.split('.')[-1].lower() in audio_format]
    """
    song = AudioSegment.from_mp3(audio_file)
    # song = AudioSegment.from_wav(audio_file)
    # ogg_version = AudioSegment.from_ogg(audio_file)
    # flv_version = AudioSegment.from_flv(audio_file)
    # mp4_version = AudioSegment.from_file(audio_file, "mp4")
    # wma_version = AudioSegment.from_file(audio_file, "wma")
    # aac_version = AudioSegment.from_file(audio_file, "aac")
    datas = get_audio_result_json(json_file)
    for data in datas:
        part = song[data[0]*1000:data[1]*1000]
        print(data[2])
        part.export('./' + data[2] + '.mp3', format='mp3')


if __name__ == '__main__':
    audio_file = './audio.mp3'
    file = './project刘爽.json'
    split_sound(audio_file, file)
