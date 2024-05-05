import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import random
from sklearn.cluster import MiniBatchKMeans
import json
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import colorspacious
st.set_page_config(page_title='PokéballPicker',page_icon='cover.jpg',layout='centered')
clusterzahl = 5
balldict = {
    'Freundesball': 'Freundesball.png',
    'Wiederball': 'Wiederball.png',
    'Tauchball': 'Tauchball.png',
    'Nestball': 'Nestball.png',
    'Timerball': 'Timerball.png',
    'Hyperball': 'Hyperball.png',
    'Flottball': 'Flottball.png',
    'Heilball': 'Heilball.png',
    'Levelball': 'Levelball.png',
    'Traumball': 'Traumball.png',
    'Sympaball': 'Sympaball.png',
    'Superball': 'Superball.png',
    'Luxusball': 'Luxusball.png',
    'Mondball': 'Mondball.png',
    'Koederball': 'Koederball.png',
    'Schwerball': 'Schwerball.png',
    'Premierball': 'Premierball.png',
    'Pokeball': 'Pokeball.png',
    'Finsterball': 'Finsterball.png',
    'Netzball': 'Netzball.png',
    'Meisterball': 'Meisterball.png',
    'Ultraball': 'Ultraball.png',
    'Turboball':'Turboball.png'
}


def plot_horizontal_stacked_bar(df):
    # DataFrame sortieren
    df = df.sort_values(by='anteil', ascending=False).reset_index(drop=True)

    # Plot initialisieren
    fig, ax = plt.subplots(figsize=(10, 2))  # Reduzierte Höhe für ein schlankes Diagramm

    # Startposition für jeden Balken
    starts = [0]
    for idx in range(1, len(df)):
        starts.append(starts[idx - 1] + df.iloc[idx - 1]['anteil'])

    # Balken hinzufügen
    for idx, row in df.iterrows():
        ax.barh(0, row['anteil'], left=starts[idx], color=tuple(val / 255 for val in row['farbe']), edgecolor='black')

    # Diagramm konfigurieren
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_frame_on(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.tight_layout()
    return plt

def calculate_color_distance(pokemondf, balldf):
    total_distance = 0
    
    # Durchlaufe jede Farbkombination der beiden DataFrames
    for _, poke_row in pokemondf.iterrows():
        for _, ball_row in balldf.iterrows():
            # Konvertiere RGB-Tupel in CIELAB unter Verwendung der CIECAM02 UCS (Uniform Colour Space)
            poke_color_lab = colorspacious.cspace_convert(poke_row['farbe'], "sRGB255", "CIELab")
            ball_color_lab = colorspacious.cspace_convert(ball_row['farbe'], "sRGB255", "CIELab")
            
            # Berechne die CIEDE2000 Distanz
            color_distance = colorspacious.deltaE(poke_color_lab, ball_color_lab, input_space="CIELab")
            
            # Gewichte die Distanz mit dem Produkt der Anteile und summiere auf
            total_distance += color_distance * poke_row['anteil'] * ball_row['anteil']

    return total_distance

# Funktion zum Laden aller Pokéball-Bilder beim Start der App
if 'ball_images' not in st.session_state:
    st.session_state['ball_images'] = {}
    for key, value in balldict.items():
        img = Image.open(value).convert('RGBA')
        st.session_state['ball_images'][key] = img

if 'ballvalues' not in st.session_state:
    # Laden des JSON-Strings aus der Datei
    with open('ballvalues.json', 'r') as f:
        json_string = f.read()

    # Konvertieren des JSON-Strings zurück in ein Dictionary
    loaded_json = json.loads(json_string)

    # Konvertieren jedes JSON-Strings im Dictionary zurück in ein DataFrame
    st.session_state['ballvalues'] = {key: pd.read_json(df_json) for key, df_json in loaded_json.items()}


@st.cache_data(max_entries=5)
def load_image(image_path):
    response = requests.get(image_path)
    response.raise_for_status()  # Stellt sicher, dass die Anfrage erfolgreich war
    img = Image.open(BytesIO(response.content)).convert('RGBA')
    return img

def get_main_colors(img, n_colors):   
    data = np.array([pixel[:3] for pixel in img.getdata() if pixel[3] != 0], np.uint8)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(data)
    unique_labels, labelcount = np.unique(labels, return_counts=True)
    labeldf = pd.DataFrame({'cluster':unique_labels,'anzahl':labelcount})
    labeldf['anteil'] = labeldf.anzahl / labeldf.anzahl.sum()
    colors = [tuple(map(int, center)) for center in centers]
    labeldf['farbe'] = [colors[cluster] for cluster in unique_labels]
    return labeldf


if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_excel('pokemondf.xlsx')

with open('color_data.json', 'r') as file:
    balls_colors = json.load(file)
df = st.session_state['data'].copy()

st.title('Pokeball-Picker')
st.sidebar.write('In dieser App kannst du ein Pokémon auswählen und erhältst anschließend Vorschläge für die zu dem Sprite passenden Bälle! Solltest du mit der Darstellung des Sprites unzufrieden sein, kannst du im unteren Teil der App auch ein eigenes Bild hochladen und dir entsprechende Bälle empfehlen lassen. Die Basis für die Empfehlungen bilden die "Hauptfarben" des Pokémon, die mit den Hauptfarben der Bälle verglichen werden.')


pokemon_choice = st.selectbox("Wähle ein Pokémon:", df['germanname'].unique())
row = df[df['germanname'] == pokemon_choice].iloc[0]
is_shiny = st.checkbox("Shiny-Version auswählen")
normal_sprite_path = row.sprite
shiny_sprite_path = row.shiny_sprite
chosen_sprite_path = shiny_sprite_path if is_shiny else normal_sprite_path
pokemon_sprite = load_image(chosen_sprite_path)
pokemon_percent = get_main_colors(pokemon_sprite, n_colors=clusterzahl)

matches = {}
for key,value in st.session_state['ballvalues'].items():
    dist = calculate_color_distance(pokemon_percent,st.session_state['ballvalues'][key])
    matches[key] = dist


# Sortiere die Bälle nach ihrem Match-Score und extrahiere die Namen der besten drei Bälle
best_three_balls = [name for name, score in sorted(matches.items(), key=lambda item: item[1])[:3]]


col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
with col1:
    st.subheader('Sprite')
    st.image(chosen_sprite_path, use_column_width=True)
    st.pyplot(plot_horizontal_stacked_bar(pokemon_percent))

with col2:
    st.subheader('Best Balls \n (Dominant Color)')
    for ball in best_three_balls:
        if ball in st.session_state['ball_images']:
            st.image(st.session_state['ball_images'][ball], use_column_width=False)


# Auswahl der richtigen Spalten basierend auf 'is_shiny'
if is_shiny:
    ball_columns = ['shiny_ball_1', 'shiny_ball_2', 'shiny_ball_3']
else:
    ball_columns = ['ball_1', 'ball_2', 'ball_3']


with col3:
    st.subheader('Best Balls \n (Human choice)')
    for col in ball_columns:
        if pd.notna(row[col]):
            st.image(st.session_state['ball_images'][row[col]], use_column_width=False)



footer = """
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Human Choice is based on reddit user OracleLink <a style='display: block; text-align: center;' href="https://docs.google.com/spreadsheets/d/1bvIx7Q2Lxp7efHRrUh48WkuwirNlKardwSHVz_R8kA0/edit#gid=1553039354" target="_blank">And his spreadsheet</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
