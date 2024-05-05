import colorspacious
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    'Turboball':'Turboball.png',
    'Turnierball':'Turnierball.png',
    'Safariball':'Safariball.png'
}