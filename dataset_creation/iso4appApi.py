import requests
import time
from typing import List, Dict

BASE_URL = "https://api.iso4app.com"

def login(username: str, password: str) -> Dict:
    """
    Effettua il login e restituisce la risposta JSON con il token di accesso.
    """
    url = f"{BASE_URL}/user/login"
    login_info = {
        "username": username,
        "password": password,
        "clientVersion": "1.0.0"
    }
    
    response = requests.post(
        url,
        json=login_info,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )
    
    if response.ok:
        return response.json()
    else:
        raise Exception(f"Errore login: {response.text}")

def get_population(auth_token: str, coords: List[Dict], category: int) -> Dict:
    """
    Ottiene i dati della popolazione per un poligono specifico.
    """
    url = f"{BASE_URL}/indicator"
    
    # Converti le coordinate nel formato richiesto
    txt_coords = ",".join([f"{point['lat']} {point['lng']}" for point in coords])
    
    parametri = {
        "polygon": txt_coords,
        "category": category,
        "returnOnlyValue": True
    }
    
    response = requests.post(
        url,
        json=parametri,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {auth_token}"
        }
    )
    
    if response.ok:
        return response.json()
    else:
        raise Exception(f"Errore get_population: {response.text}")

def run_test():
    """
    Funzione principale che esegue il test.
    """
    print("Inizio test...\n")
    
    # Login
    response = login(username="federicom", password="federico0504!")
    access_token = response["accessToken"]
    print(f"Login effettuato con successo. Token ottenuto.\n")
    
    # Definisci il rettangolo
    rectangle = [
        {"lat": 45.1404391, "lng": 10.0223644},
        {"lat": 45.1404391, "lng": 10.0439929},
        {"lat": 45.1295404, "lng": 10.0439929},
        {"lat": 45.1295404, "lng": 10.0223644}
    ]
    
    # Esegui 4 chiamate con intervallo di 2 secondi
    max_volte = 4
    for contatore in range(1, max_volte + 1):
        try:
            result = get_population(
                auth_token=access_token,
                coords=rectangle,
                category=1000
            )
            print(f"Chiamata {contatore} popolazione: {result['v']}")
        except Exception as e:
            print(f"Chiamata {contatore} errore: {e}")
        
        # Attendi 2 secondi prima della prossima chiamata (tranne all'ultima)
        if contatore < max_volte:
            time.sleep(2)
    
    print("\nTimer completato e fermato.")

if __name__ == "__main__":
    run_test()