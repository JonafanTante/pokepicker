import streamlit as st
import pandas as pd
from PIL import Image
import json
import requests
from io import BytesIO, StringIO


from functions import plot_horizontal_stacked_bar, calculate_color_distance, get_main_colors, balldict

st.set_page_config(page_title='PokéballPicker',page_icon='cover.jpg',layout='centered')

clusterzahl = 5

if 'ball_images' not in st.session_state or 'ballvalues' not in st.session_state or 'data' not in st.session_state:
    st.session_state['ball_images'] = {}
    for key, value in balldict.items():
        img = Image.open(value).convert('RGBA')
        st.session_state['ball_images'][key] = img

    with open('ballvalues.json', 'r') as f:
        json_string = f.read()
    # Konvertieren des JSON-Strings zurück in ein Dictionary
    loaded_json = json.loads(json_string)
    # Konvertieren jedes JSON-Strings im Dictionary zurück in ein DataFrame
    st.session_state['ballvalues'] = {key: pd.read_json(StringIO(df_json)) for key, df_json in loaded_json.items()}

    st.session_state['data'] = pd.read_excel('pokemondf.xlsx')

@st.cache_data(max_entries=5)
def load_image(image_path):
    response = requests.get(image_path)
    response.raise_for_status()  # Stellt sicher, dass die Anfrage erfolgreich war
    img = Image.open(BytesIO(response.content)).convert('RGBA')
    return img


st.title('Pokeball-Picker')
df = st.session_state['data'].copy()

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
color: white; /* Textfarbe auf Weiß geändert */
text-align: center;
text-shadow: -1px -1px 0 #000,  
              1px -1px 0 #000,
             -1px  1px 0 #000,
              1px  1px 0 #000; /* Schwarze Umrandung */
}
</style>
<div class="footer">
<p>Human Choice is based on reddit user OracleLink <a style='display: block; text-align: center;' href="https://docs.google.com/spreadsheets/d/1bvIx7Q2Lxp7efHRrUh48WkuwirNlKardwSHVz_R8kA0/edit#gid=1553039354" target="_blank">And his spreadsheet</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
