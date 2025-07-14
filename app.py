import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import albumentations as A

# Ładowanie modelu
model = tf.keras.models.load_model('f_mnist_model_old.h5')

# Nazwy klas w Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Inicjalizacja aplikacji Dash
app = Dash(__name__)

# Layout aplikacji
app.layout = html.Div([
    html.H1('Fashion MNIST - Klasyfikator ubrań', style={'textAlign': 'center'}),
    
    # Sekcja wgrywania pliku
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Przeciągnij i upuść lub ',
                html.A('wybierz plik')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
    ]),
    
    # Sekcja wyświetlania obrazu i predykcji
    html.Div([
        # Wyświetlanie obrazu
        html.Div(id='output-image-upload', style={'textAlign': 'center'}),
        # Wyświetlanie predykcji
        html.Div(id='prediction-output', style={'textAlign': 'center', 'margin': '20px'})
    ])
])

def preprocess_image(contents):
    """Przetwarza wgrany obraz do formatu wymaganego przez model."""
    # Dekodowanie zawartości obrazu
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Konwersja do obrazu PIL i następnie do numpy array
    img = Image.open(io.BytesIO(decoded)).convert('L')  # Konwersja do skali szarości
    img_array = np.array(img)
    
    # Definiowanie transformacji Albumentations
    transform = A.Compose([
        A.InvertImg(p=1),
        A.LongestMaxSize(max_size=26),  # Zmniejszamy do 26x26, zostawiając miejsce na padding
        # A.Resize(height=28, width=28, p=1),
        A.PadIfNeeded(
            min_height=28, 
            min_width=28, 
            border_mode=0,  # Czarne krawędzie
            position='center'
        )
    ])
    
    # Aplikowanie transformacji
    transformed = transform(image=img_array)
    img_array = transformed['image']
    
    # Normalizacja
    img_array = img_array / 255.0
    
    return img_array

@callback(
    [Output('output-image-upload', 'children'),
     Output('prediction-output', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return html.Div('Wgraj obraz...'), ''
    
    try:
        # Przetworzenie obrazu
        img_array = preprocess_image(contents)
        
        # Wykonanie predykcji
        predictions = model.predict(np.expand_dims(img_array, axis=0), verbose=1)
        predicted_class = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class]
        confidence = predictions[0][predicted_class] * 100
        
        # Przygotowanie wyświetlenia obrazu
        fig = px.imshow(img_array, binary_string=True)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_visible=False,
            yaxis_visible=False
        )
        
        return [
            # Wyświetlenie obrazu
            dcc.Graph(figure=fig, style={'width': '300px', 'height': '300px', 'margin': 'auto'}),
            # Wyświetlenie predykcji
            html.Div([
                html.H3(f'Przewidywana klasa: {predicted_class_name}'),
                html.P(f'Pewność: {confidence:.2f}%')
            ])
        ]
        
    except Exception as e:
        return html.Div([
            'Wystąpił błąd podczas przetwarzania pliku.',
            html.Pre(str(e))
        ]), ''

if __name__ == '__main__':
    app.run(debug=True) 