# Railway Data Analysis App

Aplikasi web untuk analisis data kereta api yang di-deploy di Railway.

## 🚀 Features

- **Analisis Ketebalan**: Visualisasi ketebalan kabel T1-T4
- **Analisis Ketinggian**: Visualisasi ketinggian jalur T1-T4  
- **Analisis Deviasi**: Visualisasi deviasi jalur T1-T4
- **Export PDF**: Generate laporan dalam format PDF
- **Database Integration**: Koneksi ke SQL Server

## 🔧 Environment Variables

Set these variables in Railway dashboard:

```bash
SECRET_KEY=your-very-secure-secret-key
DB_SERVER=your-sql-server-host
DB_PORT=1433
DB_DATABASE=KAI
DB_USERNAME=your-username
DB_PASSWORD=your-password
```

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /api/test-connection` - Test database connection
- `GET /api/databases` - Get list of databases
- `GET /api/tables/<database>` - Get tables from database
- `GET /api/lintas/<database>/<table>` - Get lintas data
- `GET /api/trip/<database>/<table>/<lintas>` - Get trip data

## 🛠 Local Development

```bash
pip install -r requirements.txt
python app.py
```

## 🚢 Deployment

Deployed automatically to Railway via GitHub integration.

## 📁 Project Structure

```
├── app.py                 # Main Flask application
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── nixpacks.toml       # Nixpacks configuration
└── README.md           # This file
```
