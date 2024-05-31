import requests

# URL del servidor Flask
url = 'http://127.0.0.1:5000/subir'

# Ruta del archivo a subir
#file_path = 'C:/Users/villa/OneDrive/Pictures/Screenshots/Captura de pantalla 2023-12-11 171738.png'
file_path = 'C:/Users/villa/IA/data/170696219-f68699c6-1e82-46bf-aaed-8e2fc3fa5f7b.jpg'

# Abrir el archivo en modo binario
with open(file_path, 'rb') as file:
    # Crear un diccionario de archivos para la solicitud
    files = {'file': file}
    
    # Hacer la solicitud POST
    response = requests.post(url, files=files)
    
    # Imprimir la respuesta del servidor
    print(response.json())
