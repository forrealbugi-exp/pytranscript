"""Diagnostyka Ollama – uruchom, aby sprawdzić co działa, a co nie."""
import socket
import shutil
import subprocess
import sys

print("=" * 60)
print("DIAGNOSTYKA OLLAMA")
print("=" * 60)

# 1. Czy pakiet Python 'ollama' jest zainstalowany?
print("\n[1] Pakiet Python 'ollama':")
try:
    import ollama
    print(f"    ✅ Zainstalowany (wersja: {getattr(ollama, '__version__', '?')})")
except ImportError:
    print("    ❌ BRAK. Uruchom: pip install ollama")

# 2. Czy komenda 'ollama' jest dostępna w PATH?
print("\n[2] Komenda 'ollama' w PATH:")
ollama_path = shutil.which("ollama")
if ollama_path:
    print(f"    ✅ Znaleziona: {ollama_path}")
    try:
        result = subprocess.run(["ollama", "--version"],
                                capture_output=True, text=True, timeout=5)
        print(f"    Wersja: {result.stdout.strip() or result.stderr.strip()}")
    except Exception as e:
        print(f"    ⚠️  Błąd: {e}")
else:
    print("    ❌ NIE ZNALEZIONO. Ollama (aplikacja) nie jest zainstalowana.")
    print("    Pobierz: https://ollama.com/download")

# 3. Czy serwer Ollama nasłuchuje na porcie 11434?
print("\n[3] Serwer Ollama (port 11434):")
try:
    with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
        print("    ✅ Serwer odpowiada – Ollama działa w tle.")
except (OSError, ConnectionRefusedError):
    print("    ❌ Serwer NIE odpowiada.")
    print("    Uruchom w osobnym oknie: ollama serve")
    print("    lub otwórz aplikację Ollama (ikona w zasobniku systemowym).")

# 4. Lista modeli (tylko jeśli serwer działa)
print("\n[4] Lista modeli:")
try:
    with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
        try:
            result = subprocess.run(["ollama", "list"],
                                    capture_output=True, text=True, timeout=10)
            out = result.stdout.strip()
            if out:
                print(out)
            else:
                print("    ⚠️  Pusto. Pobierz model: ollama pull gemma3")
        except Exception as e:
            print(f"    ⚠️  Błąd: {e}")
except (OSError, ConnectionRefusedError):
    print("    ⏭️  Pominięte – serwer nie działa.")

print("\n" + "=" * 60)