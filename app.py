from O365 import Account
from flask import Flask, render_template_string, request, redirect

# --- 1. CONFIGURACIÓN ---
client_id = '22167937-636a-4a36-a14f-99101c51fbd7'
client_secret = '' 
tenant_id = 'common'

credentials = (client_id, client_secret)
account = Account(credentials, tenant_id=tenant_id)
redirect_uri = 'https://login.microsoftonline.com/common/oauth2/nativeclient'

# Permisos para Leer y Enviar
scopes = ['https://graph.microsoft.com/Mail.Read', 'https://graph.microsoft.com/Mail.Send']

if not account.is_authenticated:
    account.authenticate(scopes=scopes, redirect_uri=redirect_uri)

app = Flask(__name__)

# --- TEMPLATE HTML (Con Pestañas JS y Tailwind) ---
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Outlook Web App Custom</title>
</head>
<body class="bg-slate-900 text-slate-200 p-4 md:p-8 min-h-screen">
    <div class="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        
        <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl h-fit">
            <h2 class="text-xl font-bold text-blue-400 mb-4">Redactar Mensaje</h2>
            <form action="/enviar" method="POST" class="space-y-4">
                <div>
                    <label class="block text-xs uppercase text-slate-400 mb-1">Para:</label>
                    <input type="email" name="destinatario" required class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-xs uppercase text-slate-400 mb-1">Asunto:</label>
                    <input type="text" name="asunto" required class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-xs uppercase text-slate-400 mb-1">Mensaje:</label>
                    <textarea name="cuerpo" rows="6" required class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-sm focus:border-blue-500 outline-none"></textarea>
                </div>
                <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 rounded transition">Enviar Correo</button>
            </form>
        </div>

        <div class="md:col-span-2">
            
            <div class="flex space-x-6 mb-4 border-b border-slate-700">
                <button onclick="showTab('recibidos')" id="tab-recibidos" class="pb-2 border-b-2 border-blue-500 text-blue-400 font-bold text-lg transition">Recibidos</button>
                <button onclick="showTab('enviados')" id="tab-enviados" class="pb-2 border-b-2 border-transparent text-slate-400 hover:text-slate-200 font-bold text-lg transition">Enviados</button>
            </div>

            <div id="content-recibidos" class="bg-slate-800 rounded-xl border border-slate-700 shadow-xl divide-y divide-slate-700">
                {% for msg in recibidos %}
                <div class="p-4 hover:bg-slate-750 transition">
                    <div class="flex justify-between text-xs text-slate-400 mb-1">
                        <span class="font-bold text-slate-300">De: {{ msg.sender.address if msg.sender else 'Desconocido' }}</span>
                        <span>{{ msg.received.strftime('%d-%m-%Y %H:%M') }}</span>
                    </div>
                    <p class="font-medium text-slate-100 text-lg">{{ msg.subject }}</p>
                    <details class="mt-2 group">
                        <summary class="text-xs text-blue-400 cursor-pointer font-medium hover:text-blue-300">Leer correo ▼</summary>
                        <div class="mt-3 bg-white text-black p-4 rounded text-sm overflow-auto max-h-60 shadow-inner border border-slate-300">
                            {{ msg.body | safe }}
                        </div>
                    </details>
                </div>
                {% else %}
                <div class="p-8 text-center text-slate-500 italic">No hay correos en tu bandeja de entrada.</div>
                {% endfor %}
            </div>

            <div id="content-enviados" class="hidden bg-slate-800 rounded-xl border border-slate-700 shadow-xl divide-y divide-slate-700">
                {% for msg in enviados %}
                <div class="p-4 hover:bg-slate-750 transition">
                    <div class="flex justify-between text-xs text-slate-400 mb-1">
                        <span class="font-bold text-slate-300">Para: {{ msg.to[0].address if msg.to else 'Desconocido' }}</span>
                        <span>{{ msg.received.strftime('%d-%m-%Y %H:%M') }}</span>
                    </div>
                    <p class="font-medium text-slate-100 text-lg">{{ msg.subject }}</p>
                    <details class="mt-2 group">
                        <summary class="text-xs text-blue-400 cursor-pointer font-medium hover:text-blue-300">Leer correo ▼</summary>
                        <div class="mt-3 bg-white text-black p-4 rounded text-sm overflow-auto max-h-60 shadow-inner border border-slate-300">
                            {{ msg.body | safe }}
                        </div>
                    </details>
                </div>
                {% else %}
                <div class="p-8 text-center text-slate-500 italic">No tienes correos enviados recientemente.</div>
                {% endfor %}
            </div>

        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Ocultar ambas bandejas
            document.getElementById('content-recibidos').classList.add('hidden');
            document.getElementById('content-enviados').classList.add('hidden');
            
            // Reiniciar el estilo visual de los botones (desactivados)
            const styleInactive = "pb-2 border-b-2 border-transparent text-slate-400 hover:text-slate-200 font-bold text-lg transition";
            document.getElementById('tab-recibidos').className = styleInactive;
            document.getElementById('tab-enviados').className = styleInactive;
            
            // Mostrar la bandeja seleccionada
            document.getElementById('content-' + tabName).classList.remove('hidden');
            
            // Activar el estilo visual del botón clickeado
            const styleActive = "pb-2 border-b-2 border-blue-500 text-blue-400 font-bold text-lg transition";
            document.getElementById('tab-' + tabName).className = styleActive;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    mailbox = account.mailbox()
    
    # 1. Obtener Bandeja de Entrada (Inbox)
    inbox = mailbox.inbox_folder()
    recibidos_data = inbox.get_messages(limit=10)
    
    # 2. Obtener Correos Enviados (Sent Items)
    sent_folder = mailbox.sent_folder()
    enviados_data = sent_folder.get_messages(limit=10)
    
    return render_template_string(HTML_LAYOUT, recibidos=recibidos_data, enviados=enviados_data)

@app.route('/enviar', methods=['POST'])
def enviar():
    m = account.new_message()
    m.to.add(request.form['destinatario'])
    m.subject = request.form['asunto']
    m.body = request.form['cuerpo']
    m.send()
    # Al terminar de enviar, recarga la página principal
    return redirect('/')

if __name__ == '__main__':
    app.run(port=5000, debug=False)