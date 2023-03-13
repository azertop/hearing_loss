import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import wave



st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar : 
    p=st.slider("Pourcentage de surdité",min_value=0,max_value=100,step=1)
    p*=0.01
    b=st.checkbox("Je souhaite importer un son")
    if b : 
        f=st.file_uploader("Importer un fichier")
        if f is not None :
            f=open(f,'rb')
            with wave.open(f) as audio :
                y,sr=librosa.load(audio)
        else : 
            y,sr=0
    else : 
        filename = "file.wav"
        f=open(filename,'rb')
        y,sr = librosa.load(filename)
    valider = st.button("Valider")


if valider:
    fft_signal = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(y), 1/sr)
    a=np.random.choice(len(fft_signal), size=int(len(fft_signal)*p), replace=False)
    fft_signal[a]=0
    y_filtered = np.fft.ifft(fft_signal).real
    plt.plot(fft_signal)
    plt.show()
    y_norm = librosa.util.normalize(y_filtered)
    sf.write('output_file.wav',y_norm, sr,'PCM_24')
    audio_file = open('output_file.wav','rb')
    st.write("Audio d'origine")
    st.audio(f)
    st.write("Audio après transformation")
    st.audio(audio_file)
    col1,col2=st.columns(2)
    with col1:
        st.header('Audio de base')
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)
    with col2:
        st.header('Audio transformé')
        fig2, ax2 = plt.subplots()
        ax2.plot(y_norm)
        st.pyplot(fig2)
