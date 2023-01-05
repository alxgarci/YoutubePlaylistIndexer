import concurrent.futures
import datetime
import os
import sys
import time

import pandas
from pytube import Playlist
from pytube import YouTube
import speech_recognition as sp
from pydub import AudioSegment
from pydub.utils import make_chunks
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# py -m pip install SpeechRecognition pydub

AUDIO_FOLDER = "data/audio"
TEXT_FOLDER = "data/text"
CSV_FILE = TEXT_FOLDER + "/" + "recognized.csv"
CSV_FILE_CHUNKS = TEXT_FOLDER + "/" + "recognized_chunks.csv"
CSV_EXPORTED_TOKENIZED = TEXT_FOLDER + "/" + "recognized_tokenized.csv"
VIDEO_URL = "https://www.youtube.com/watch?v="
CHUNK_TIME = 8000
CHAR_TERMINAL_ANCHO = 96


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def log_print(ty, msg):
    if ty == "info":
        print("[+] " + msg)
    elif ty == "system":
        print("[S] " + msg)
    elif ty == "error":
        print("[ERROR] " + msg)


# Metodo muy util obtenido de:
# https://stackoverflow.com/questions/3173320/
# text-progress-bar-in-terminal-with-block-characters/13685020
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def read_input():
    input_st = input(">>: ")
    input_st.strip()
    return input_st


def print_intro():
    cls()

    logo = (
        "██╗   ██╗████████╗        ███████╗██╗███╗   ██╗██████╗ ███████╗██████╗ \n"
        "╚██╗ ██╔╝╚══██╔══╝        ██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗\n"
        " ╚████╔╝    ██║           █████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝\n"
        "  ╚██╔╝     ██║           ██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗\n"
        "   ██║      ██║           ██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║\n"
        "   ╚═╝      ╚═╝           ╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝\n")

    print("#" * CHAR_TERMINAL_ANCHO)
    for line in logo.split("\n"):
        print(line.center(CHAR_TERMINAL_ANCHO))
    print("INDEXADO DE VIDEOS DE PLAYLIST A TEXTO Y BUSQUEDA DE TEXTO POR MOMENTO EXACTO".center(CHAR_TERMINAL_ANCHO))
    print("Alejandro Garcia, Jorge Garcia".center(CHAR_TERMINAL_ANCHO))
    print("#" * CHAR_TERMINAL_ANCHO)
    print()


def ask_youtube_playlist():
    log_print("system", "Introduce el enlace de la playlist de YouTube")
    pl_str = read_input()
    playlist = Playlist(pl_str)
    return playlist


################################################################################
# INICIO MULTITHREAD - VARIOS HILOS CONCURRENTES
################################################################################

def convert_all_to_wav_multithread():
    files = []
    for file in os.listdir(AUDIO_FOLDER):
        audio_file = AUDIO_FOLDER + "/" + file
        files.append(audio_file)

    total_l = len(files)
    log_print("info", "Convirtiendo a .wav los mp4 descargados")

    print_progress_bar(0, total_l, prefix='Progreso:', suffix='Convertidos', length=70)
    counter = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        for result in enumerate(executor.map(convert_file_to_wav, files)):
            counter = counter + 1
            print_progress_bar(counter, total_l, prefix='Progreso:', suffix='Convertidos', length=70)


def convert_file_to_wav(audio_file):
    formato = "mp4"
    sound = AudioSegment.from_file(audio_file, formato)
    sound.export(audio_file.replace(formato, "") + ".wav", format="wav")
    # log_print("info", audio_file + " converted to .wav")
    os.remove(audio_file)

    ################################################################################
    # INICIO DE TRANSCRIPCION POR CHUNKS EN VEZ DE VIDEO
    ################################################################################


def process_audio_multithread_ch():
    total_space = 0
    total_time = 0
    st = time.time()
    total_l = len(os.listdir(AUDIO_FOLDER))
    log_print("info", "Transcribiendo texto a partir del audio de cada video")
    print_progress_bar(0, total_l, prefix='Progreso:', suffix='Complete', length=70)

    for i, file in enumerate(os.listdir(AUDIO_FOLDER)):
        audio_file = AUDIO_FOLDER + "/" + file
        total_space = total_space + os.path.getsize(audio_file) / (1024 * 1024)
        total_time = total_time + YouTube(VIDEO_URL + file.replace(".wav", "")).length
        print_progress_bar(i + 1, total_l, prefix='Progreso:', suffix=file, length=70)
        audio_to_text_multithread_ch(audio_file)
        os.remove(audio_file)
    et = time.time()
    elapsed = et - st
    print_intro()
    log_print("system", "Se han transcrito " + '{:.2f}'.format(total_space) + " MB de audio en "
              + str(datetime.timedelta(seconds=int(elapsed))))
    log_print("system", "Son un total de " + str(datetime.timedelta(seconds=total_time))
              + " hh:mm:ss de video transcritos")


def audio_to_text_multithread_aux_ch(audio_file):
    recognizer = sp.Recognizer()
    with sp.AudioFile(audio_file) as source:
        listen = recognizer.listen(source)
    try:
        rec = recognizer.recognize_google(listen, show_all=False)
        result_string = rec
        os.remove(audio_file)
        return result_string, audio_file
    except sp.UnknownValueError:
        # print("No se reconoce el audio")
        os.remove(audio_file)
        return "", audio_file
    except sp.RequestError:
        os.remove(audio_file)
        return "", audio_file


def audio_to_text_multithread_ch(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    chunks_length = CHUNK_TIME  # Milisegundos
    chunks = make_chunks(audio, chunk_length=chunks_length)
    file_list = []
    for i, chunk in enumerate(chunks):
        chunk_name = audio_file.replace(".wav", "") + "_{0}.wav".format(i)
        chunk.export(chunk_name, format="wav")
        file_list.append(chunk_name)

    # log_print("info", "Comenzando transcripcion de " + audio_file)
    # Metodo hidden prints usado para solucionar bug con los print del metodo recognize_google
    with HiddenPrints():
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            for transcription, audio_f in executor.map(audio_to_text_multithread_aux_ch, file_list):
                if transcription:
                    save_text_to_csv_ch(transcription, audio_f)


def save_text_to_csv_ch(text, video):
    header = ["text", "chunk"]
    row = [text, video]
    if os.path.exists(CSV_FILE_CHUNKS):
        with open(CSV_FILE_CHUNKS, 'a', encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open(CSV_FILE_CHUNKS, 'w', encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

    ################################################################################
    # FIN DE TRANSCRIPCION POR CHUNKS EN VEZ DE VIDEO
    ################################################################################


################################################################################
# FIN MULTITHREAD - VARIOS HILOS CONCURRENTES
################################################################################


def playlist_to_audio(p):
    log_print("info", "Descargando los mp4 only-audio de la playlist especificada")
    print_progress_bar(0, len(p), prefix='Progreso:', suffix='Descargados', length=70)
    for i, url in enumerate(p):
        YouTube(url).streams.filter(only_audio=True).first() \
            .download(
            output_path=AUDIO_FOLDER,
            filename=url.replace(VIDEO_URL, "")
        )

        # log_print("info", "\"" + YouTube(url).title + "\" downloaded as mp4 only-audio          ")
        print_progress_bar(i + 1, len(p), prefix='Progreso:', suffix="Descargado", length=70)


def clean_all():
    for file in os.listdir(AUDIO_FOLDER):
        os.remove(AUDIO_FOLDER + "/" + file)
    for file in os.listdir(TEXT_FOLDER):
        os.remove(TEXT_FOLDER + "/" + file)


def nltk_tokenizer(text):
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stop_words]
    tokens = list(set(tokens))
    porter_stemmer = PorterStemmer()
    tk_stemmed = []
    for tk in tokens:
        tk_stemmed.append(porter_stemmer.stem(tk, to_lowercase=True))
    return tk_stemmed


def tokenize_to_pandas():
    log_print("info", "Sacando tokens de cada texto, quitando stop words y aplicando stemming")
    # recognized = pd.read_csv(CSV_FILE)
    recognized = pd.read_csv(CSV_FILE_CHUNKS)
    # recognized = recognized.drop("url")
    recognized["text"] = recognized["text"].astype(str)
    recognized["tokens"] = recognized["text"].apply(lambda x: nltk_tokenizer(x))
    log_print("info", "Mostrando head de dataframe con tokens ya procesados")
    log_print("info", "Con un total de {0} filas".format(len(recognized.index)))
    print("\n" + "+" * CHAR_TERMINAL_ANCHO)
    print(recognized.head())
    print("+" * CHAR_TERMINAL_ANCHO + "\n")
    recognized.to_csv(CSV_EXPORTED_TOKENIZED)
    log_print("info", "CSV Tokenizado exportado en " + CSV_EXPORTED_TOKENIZED)
    return recognized


def get_query_to_search(rd):
    log_print("system", "Introduce la Query a buscar")
    q_str = read_input()
    query = nltk_tokenizer(q_str)
    rd["palabras_incluidas"] = rd["tokens"].apply(lambda x: list(set(query).intersection(x)))
    rd["coincidencia"] = rd["tokens"].apply(
        lambda x: '{:.2f}%'.format((len(list(set(query).intersection(x))) / len(query)) * 100))
    rd["query_match"] = rd["palabras_incluidas"].apply(lambda x: "True" if x else "False")

    rd_aux = rd[(rd["query_match"] == "True")]
    rd_aux = rd_aux.drop("text", axis=1)
    rd_aux = rd_aux.drop("tokens", axis=1)
    rd_aux = rd_aux.drop("query_match", axis=1)
    # rd_aux["coincidencia"] = rd_aux["coincidencia"].apply(lambda x: x.replace("%", ""))
    # rd_aux = rd_aux.drop(rd_aux[rd_aux["coincidencia"].astype(float) <= 35.00].index)
    # rd_aux["coincidencia"] = rd_aux["coincidencia"].astype(str) + "%"
    rd_aux = rd_aux.sort_values("coincidencia", ascending=False)
    # rd_aux = rd_aux["coincidencia"].astype(str) + "%"
    # log_print("system", "Imprimiendo las coincidencias encontradas")
    if rd_aux.empty:
        log_print("error", "No se han encontrado coincidencias para la query \"" + q_str + "\"")
        for x in query:
            print(" - " + x)
    else:
        log_print("info", "Se han encontrado " + str(len(rd_aux.index)) + " coincidencias:")
        # print(rd_aux.to_string())
    return rd_aux


def return_exact_time(x):
    ex_time_seconds = (int(x) * CHUNK_TIME) / 1000
    return ex_time_seconds


def get_audio_exact_minutes(rd_aux):
    rd_aux["chunk"] = rd_aux["chunk"].apply(lambda x: x.replace(AUDIO_FOLDER + "/", "")
                                            .replace(".wav", ""))
    rd_finale = pd.DataFrame()
    rd_finale["url"] = rd_aux["chunk"].apply(lambda x: VIDEO_URL + x[0:11])
    rd_finale["chunk"] = rd_aux["chunk"].apply(lambda x: x[12:])
    rd_finale["time_to"] = rd_finale["chunk"].apply(lambda x: "&t={0}s".format(int(return_exact_time(x))))
    rd_finale["time"] = rd_finale["chunk"].apply(lambda x: time.strftime("%M:%S", time.gmtime(return_exact_time(x))))
    rd_finale["url"] = rd_finale["url"].astype(str) + rd_finale["time_to"].astype(str)
    rd_finale["coincidencia"] = rd_aux["coincidencia"]
    rd_finale = rd_finale.drop("time_to", axis=1)
    rd_finale = rd_finale.drop("chunk", axis=1)
    rd_finale["palabras_incluidas"] = rd_aux["palabras_incluidas"]
    rd_finale = rd_finale.sort_values("coincidencia", ascending=False)
    if rd_finale.empty:
        log_print("error", "Sin coincidencias")
    else:
        log_print("info", "Se imprimen los enlaces con el segundo en el que se encuentra la query")
        log_print("info", "Hay una precision de 0 a +{0} segundos".format(int(CHUNK_TIME / 1000)))
        print("\n" + "+" * CHAR_TERMINAL_ANCHO)
        print(rd_finale.to_string())
        print("+" * CHAR_TERMINAL_ANCHO + "\n")


def main():
    # Eliminar todos los archivos residuales de la anterior ejecucion
    clean_all()
    print_intro()

    log_print("system",
              "Dependiendo de la velocidad y threads de cada ordenador, el tiempo de procesado de la playlist\n    "
              "equivaldra a 1 Hora de contenido = 1 Minuto de procesado. Una vez tokenizados los videos,\n    "
              "el tiempo de busqueda de texto introducido sera instantaneo.")
    # Obtener todos los videos de una playlist de youtube
    playlist = ask_youtube_playlist()

    # Crear archivos mp4 de solo audio para cada video de la playlist
    playlist_to_audio(playlist)

    # Convertir los mp4 descargados a wav para poder procesarlos con la libreria SpeechRecognition
    # convert_all_to_wav()
    convert_all_to_wav_multithread()

    # Procesando los .wav con SpeechRecognition. Al ser audios largos, se dividen en audios mas pequeños ("chunks")
    # se procesan por separado y luego se une el texto para formar el texto que aparece en el video. Se usa
    # multithreading (varios hilos concurrentes) para obtener la transcripcion de cada "chunk" de forma mas rapida.
    # Era un proceso demasiado lento para ejecutarlo en un solo hilo cada vez.
    # process_audio_multithread()
    process_audio_multithread_ch()

    # Formamos el dataframe a partir del csv donde se guardaron los textos. A cada texto se le aplica
    # tokenizacion, stemming y se quitan stop words para poder buscar de forma correcta la query
    rd = tokenize_to_pandas()

    # Se procesa la query tokenizandolo, con stemming y quitando stop words y se saca la interseccion de los tokens
    # de la query con cada texto, indexando asi cada video que contiene las palabras y mostrando cuantas de ellas (en %)
    # ordenandolo por la cantidad de palabras que aparecen en el video
    while True:
        # rd = tokenize_to_pandas()
        print("#" * CHAR_TERMINAL_ANCHO)
        rd_aux = get_query_to_search(rd)
        get_audio_exact_minutes(rd_aux)


if __name__ == '__main__':
    main()
