import os
from matplotlib.backends.backend_pdf import PdfPages
from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify, session

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Use environment variable in production

# Default database configuration
DEFAULT_DB_CONFIG = {
    'server': 'localhost',
    'port': 1433,
    'database': 'KAI',
    'username': 'kai_user',
    'password': 'passwordku123'
}


@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify({'status': 'healthy', 'message': 'Application is running'})


@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    """Test database connection with provided configuration"""
    try:
        data = request.get_json()
        config = {
            'server': data.get('server', 'localhost'),
            'port': int(data.get('port', 1433)),
            'username': data.get('username', ''),
            'password': data.get('password', '')
        }

        # Test connection without specifying database (connect to master)
        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/master"
        )

        test_engine = create_engine(connection_string)
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Save configuration to session if connection successful (without database)
        session['db_config'] = config

        return jsonify({'success': True, 'message': 'Koneksi berhasil!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/databases')
def get_databases():
    """Endpoint untuk mengambil daftar semua database dari server"""
    try:
        config = session.get('db_config')
        if not config:
            return jsonify({'success': False, 'error': 'Konfigurasi database belum tersedia'})

        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/master"
        )

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name 
                FROM sys.databases 
                WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')
                ORDER BY name
            """))
            databases = [row[0] for row in result.fetchall()]
            return jsonify({'success': True, 'databases': databases})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/tables/<database_name>')
def get_tables(database_name):
    """Endpoint untuk mengambil daftar semua tabel dari database tertentu"""
    try:
        config = session.get('db_config')
        if not config:
            return jsonify({'success': False, 'error': 'Konfigurasi database belum tersedia'})

        # Add selected database to config
        config['database'] = database_name

        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/{database_name}"
        )

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' 
                AND TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """))
            tables = [row[0] for row in result.fetchall()]

            # Update session with selected database
            session['db_config'] = config

            return jsonify({'success': True, 'tables': tables})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/lintas/<database_name>/<table_name>')
def get_lintas(database_name, table_name):
    """Endpoint untuk mengambil daftar lintas berdasarkan database dan tabel yang dipilih"""
    try:
        config = session.get('db_config')
        if not config:
            return jsonify({'success': False, 'error': 'Konfigurasi database belum tersedia'})

        config['database'] = database_name
        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/{database_name}"
        )

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT DISTINCT LINTAS 
                FROM {table_name}
                WHERE LINTAS IS NOT NULL
                ORDER BY LINTAS
            """))
            lintas_list = [row[0] for row in result.fetchall()]
            return jsonify({'success': True, 'lintas': lintas_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/trip/<database_name>/<table_name>/<lintas>')
def get_trip(database_name, table_name, lintas):
    """Endpoint untuk mengambil daftar trip berdasarkan database, tabel dan lintas yang dipilih"""
    try:
        config = session.get('db_config')
        if not config:
            return jsonify({'success': False, 'error': 'Konfigurasi database belum tersedia'})

        config['database'] = database_name
        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/{database_name}"
        )

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT DISTINCT TRIP 
                FROM {table_name}
                WHERE LINTAS = '{lintas}' AND TRIP IS NOT NULL
                ORDER BY TRIP
            """))
            trip_list = [row[0] for row in result.fetchall()]
            return jsonify({'success': True, 'trips': trip_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def get_db_engine():
    """Get database engine with current configuration"""
    config = session.get('db_config', DEFAULT_DB_CONFIG)
    connection_string = (
        f"mssql+pymssql://{config['username']}:{config['password']}@{config['server']}:{config['port']}/{config.get('database', 'KAI')}"
    )
    return create_engine(connection_string)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        database_name = request.form['database_name']
        table_name = request.form['table_name']
        lintas = request.form['lintas']
        trip = request.form['trip']
        total_stasiun = int(request.form['total_stasiun'])
        analysis_type = request.form['analysis_type']  # Ambil jenis analisis

        # Update session with selected database
        config = session.get('db_config', DEFAULT_DB_CONFIG)
        config['database'] = database_name
        session['db_config'] = config

        stasiun_info = []
        for i in range(total_stasiun):
            nama = request.form[f'stasiun_nama_{i}']
            km = int(request.form[f'stasiun_km_{i}'])
            m = int(request.form[f'stasiun_m_{i}'])
            stasiun_info.append({'nama': nama, 'km': km, 'm': m})

        try:
            # Panggil fungsi berdasarkan jenis analisis
            if analysis_type == 'ketebalan':
                pdf_path = proses_data_ketebalan(
                    table_name, lintas, trip, stasiun_info)
            elif analysis_type == 'ketinggian':
                pdf_path = proses_data_ketinggian(
                    table_name, lintas, trip, stasiun_info)
            elif analysis_type == 'deviasi':
                pdf_path = proses_data_deviasi(
                    table_name, lintas, trip, stasiun_info)
            else:
                raise ValueError(
                    f"Jenis analisis '{analysis_type}' tidak dikenali")

            pdf_name_only = os.path.basename(pdf_path)
            return redirect(url_for('hasil', filename=pdf_name_only, analysis_type=analysis_type))
        except Exception as e:
            return render_template('error.html', message=str(e))

    return render_template('form.html')


@app.route('/hasil/<filename>')
def hasil(filename):
    analysis_type = request.args.get('analysis_type', 'ketebalan')
    return render_template('result.html', pdf_name=filename, analysis_type=analysis_type)


def proses_data_ketebalan(TABLE_NAME, LINTAS_NAME, TRIP_NAME, stasiun_info):
    """Fungsi untuk memproses data ketebalan kabel"""
    start_km = stasiun_info[0]['km']
    start_m = stasiun_info[0]['m']
    end_km = stasiun_info[-1]['km']
    end_m = stasiun_info[-1]['m']

    # Menggunakan engine yang sudah dikonfigurasi
    engine = get_db_engine()

    with engine.connect() as conn:
        df = pd.read_sql(
            f"""
            SELECT LOKASI_KM, LOKASI_M,
                   KETEBALAN_T1, KETEBALAN_T2,
                   KETEBALAN_T3, KETEBALAN_T4,
                   LINTAS, JALUR, id, TW, TRIP
            FROM {TABLE_NAME}
            WHERE LINTAS = '{LINTAS_NAME}' AND TRIP = '{TRIP_NAME}'
            ORDER BY id ASC
            """,
            con=conn
        )

        df = df.dropna(subset=['LOKASI_KM', 'LOKASI_M'])
        df['LOKASI_M'] = df['LOKASI_M'].astype(str).str.replace(',', '.')
        df['LOKASI_M'] = pd.to_numeric(df['LOKASI_M'], errors='coerce')
        df = df.dropna(subset=['LOKASI_M'])
        df['TW'] = df['TW'].astype(str)
        df = df.reset_index(drop=True)

        # Ambil informasi JALUR dan LINTAS dari data pertama
        jalur_info = df['JALUR'].iloc[0] if not df.empty else ''
        lintas_info = df['LINTAS'].iloc[0] if not df.empty else LINTAS_NAME

        # Ubah ke satuan meter agar mudah dibandingkan
        def to_meter(km, m): return km * 1000 + m
        start = to_meter(start_km, start_m)
        end = to_meter(end_km, end_m)
        df['LOKASI_TOTAL_M'] = df['LOKASI_KM'] * 1000 + df['LOKASI_M']

        is_mundur = start > end
        if is_mundur:
            df_filtered = df[(df['LOKASI_TOTAL_M'] <= start)
                             & (df['LOKASI_TOTAL_M'] >= end)]
        else:
            df_filtered = df[(df['LOKASI_TOTAL_M'] >= start)
                             & (df['LOKASI_TOTAL_M'] <= end)]

        if df_filtered.empty:
            raise ValueError("Tidak ada data dalam range KM-M yang dipilih.")

        df = df_filtered.sort_values(
            by=['LOKASI_TOTAL_M'], ascending=not is_mundur).reset_index(drop=True)

        available_columns = ['KETEBALAN_T1',
                             'KETEBALAN_T2', 'KETEBALAN_T3', 'KETEBALAN_T4']
        column_mapping = {
            'KETEBALAN_T1': ('T1', 'blue'),
            'KETEBALAN_T2': ('T2', 'green'),
            'KETEBALAN_T3': ('T3', 'orange'),
            'KETEBALAN_T4': ('T4', 'brown')
        }

        output_folder = "WEB_KETEBALAN_PER_100M"
        os.makedirs(output_folder, exist_ok=True)
        
        # SELALU URUTKAN KM DARI KECIL KE BESAR UNTUK HALAMAN PDF
        unique_km_order = sorted(df['LOKASI_KM'].drop_duplicates().tolist())
        
        output_pdf = f"{output_folder}/{TABLE_NAME}_{LINTAS_NAME}_{TRIP_NAME}_ketebalan_per100m_stasiun.pdf"

        with PdfPages(output_pdf) as pdf:
            for km in unique_km_order:
                chunk = df[df['LOKASI_KM'] == km]
                chunk = chunk[(chunk['LOKASI_M'] >= 0) &
                              (chunk['LOKASI_M'] <= 1000)]
                if chunk.empty:
                    continue

                fig, axs = plt.subplots(
                    nrows=1, ncols=4, figsize=(11.69, 8.27), sharey=True)
                unique_tw = sorted(chunk['TW'].dropna().unique())
                color_list = ['blue', 'red', 'green', 'orange', 'purple']
                tw_colors = {tw: color_list[i % len(
                    color_list)] for i, tw in enumerate(unique_tw)}

                for idx, col in enumerate(available_columns):
                    label, _ = column_mapping[col]
                    ax = axs[idx]

                    for tw in unique_tw:
                        color = tw_colors.get(tw, 'gray')
                        data_tw = chunk[chunk['TW'] == tw]
                        valid_data = data_tw[['LOKASI_M', col]].dropna()
                        if valid_data.empty:
                            continue

                        # Gunakan LOKASI_M langsung tanpa transformasi
                        valid_data['PLOT_Y'] = valid_data['LOKASI_M']
                        valid_data = valid_data.sort_values(by='LOKASI_M')

                        segments = []
                        current_segment = [valid_data.iloc[0]]
                        for j in range(1, len(valid_data)):
                            curr = valid_data.iloc[j]
                            prev = valid_data.iloc[j - 1]
                            if abs(curr['LOKASI_M'] - prev['LOKASI_M']) <= 1:
                                current_segment.append(curr)
                            else:
                                if len(current_segment) > 1:
                                    segments.append(
                                        pd.DataFrame(current_segment))
                                current_segment = [curr]
                        if len(current_segment) > 1:
                            segments.append(pd.DataFrame(current_segment))

                        for seg in segments:
                            ax.plot(seg[col], seg['PLOT_Y'], linestyle='-', linewidth=0.5,
                                    color=color, label=f'TW {tw}' if idx == 0 else None)

                        min_idx = valid_data[col].idxmin()
                        max_idx = valid_data[col].idxmax()
                        min_val = valid_data.loc[min_idx, col]
                        max_val = valid_data.loc[max_idx, col]
                        min_y = valid_data.loc[min_idx, 'PLOT_Y']
                        max_y = valid_data.loc[max_idx, 'PLOT_Y']
                        ax.text(min_val - 1, min_y, f"{min_val:.1f}", fontsize=6,
                                color=color, va='center', ha='right', weight='bold')
                        ax.text(max_val + 1, max_y, f"{max_val:.1f}", fontsize=6,
                                color=color, va='center', ha='left', weight='bold')

                    for stasiun in stasiun_info:
                        if stasiun['km'] == km:
                            y_pos = stasiun['m']  # Gunakan nilai meter langsung
                            ax.axhline(y=y_pos, color='purple',
                                       linestyle='--', linewidth=1)
                            ax.text(
                                132, y_pos, stasiun['nama'], fontsize=7, va='center', ha='left', color='purple')

                    ax.axvline(x=81, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=123.4, color='gray',
                               linestyle='--', linewidth=0.8)

                    ax.set_title(label, fontsize=10)
                    ax.set_xlim(50, 130)

                    # Atur Y axis dari atas 0 ke bawah 1000
                    ax.set_ylim(1000, 0)  # Balik urutan: dari 1000 ke 0
                    y_ticks = range(0, 1001, 100)
                    y_labels = [f"{m}" for m in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels, fontsize=8)

                    ax.grid(True, axis='y', which='major', alpha=0.3)
                    ax.set_xlabel("Ketebalan", fontsize=8)
                    if idx == 0:
                        ax.set_ylabel("LOKASI_M", fontsize=8)

                arah = "Mundur" if is_mundur else "Maju"
                fig.suptitle(
                    f"KM {int(km)} - Ketebalan T1–T4 per 100m per TW | {jalur_info} - {lintas_info} ", 
                    fontsize=12, y=0.96)

                tw_labels = [f'TW {tw}' for tw in unique_tw]
                tw_handles = [plt.Line2D(
                    [0], [0], color=tw_colors[tw], linewidth=2) for tw in unique_tw]
                fig.legend(handles=tw_handles, labels=tw_labels, loc='upper center',
                           bbox_to_anchor=(0.5, 0.92), ncol=len(tw_handles), fontsize=9, frameon=False)

                plt.tight_layout(rect=[0, 0.02, 1, 0.90])
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        return output_pdf


def proses_data_ketinggian(TABLE_NAME, LINTAS_NAME, TRIP_NAME, stasiun_info):
    """Fungsi untuk memproses data ketinggian jalur"""
    start_km = stasiun_info[0]['km']
    start_m = stasiun_info[0]['m']
    end_km = stasiun_info[-1]['km']
    end_m = stasiun_info[-1]['m']

    # Menggunakan engine yang sudah dikonfigurasi
    engine = get_db_engine()

    with engine.connect() as conn:
        df = pd.read_sql(
            f"""
            SELECT LOKASI_KM, LOKASI_M,
                   KETINGGIAN_T1, KETINGGIAN_T2,
                   KETINGGIAN_T3, KETINGGIAN_T4,
                   LINTAS, JALUR, id, TW, TRIP
            FROM {TABLE_NAME}
            WHERE LINTAS = '{LINTAS_NAME}' AND TRIP = '{TRIP_NAME}'
            ORDER BY id ASC
            """,
            con=conn
        )

        df = df.dropna(subset=['LOKASI_KM', 'LOKASI_M'])
        df['LOKASI_M'] = df['LOKASI_M'].astype(str).str.replace(',', '.')
        df['LOKASI_M'] = pd.to_numeric(df['LOKASI_M'], errors='coerce')
        df = df.dropna(subset=['LOKASI_M'])
        df['TW'] = df['TW'].astype(str)
        df = df.reset_index(drop=True)

        # Ambil informasi JALUR dan LINTAS dari data pertama
        jalur_info = df['JALUR'].iloc[0] if not df.empty else ''
        lintas_info = df['LINTAS'].iloc[0] if not df.empty else LINTAS_NAME

        # === DEBUG ARAH INPUT ===
        is_mundur_input = (start_km > end_km) or (
            start_km == end_km and start_m > end_m)

        # === FILTER SESUAI RANGE ===
        if is_mundur_input:
            df_filtered = df[
                ((df['LOKASI_KM'] < start_km) & (df['LOKASI_KM'] > end_km)) |
                ((df['LOKASI_KM'] == start_km) & (df['LOKASI_M'] <= start_m)) |
                ((df['LOKASI_KM'] == end_km) & (df['LOKASI_M'] >= end_m))
            ]
        else:
            df_filtered = df[
                ((df['LOKASI_KM'] > start_km) & (df['LOKASI_KM'] < end_km)) |
                ((df['LOKASI_KM'] == start_km) & (df['LOKASI_M'] >= start_m)) |
                ((df['LOKASI_KM'] == end_km) & (df['LOKASI_M'] <= end_m))
            ]

        if df_filtered.empty:
            raise ValueError("Tidak ada data dalam range KM-M yang dipilih.")

        df = df_filtered.copy()

        # === SORT DATA SESUAI ARAH ===
        df = df.sort_values(
            by=['LOKASI_KM', 'LOKASI_M'],
            ascending=[not is_mundur_input, not is_mundur_input]
        ).reset_index(drop=True)

        first_km_in_data = df.iloc[0]['LOKASI_KM']
        last_km_in_data = df.iloc[-1]['LOKASI_KM']
        is_mundur = first_km_in_data > last_km_in_data

        available_columns = ['KETINGGIAN_T1',
                             'KETINGGIAN_T2', 'KETINGGIAN_T3', 'KETINGGIAN_T4']
        column_mapping = {
            'KETINGGIAN_T1': ('T1', 'blue'),
            'KETINGGIAN_T2': ('T2', 'green'),
            'KETINGGIAN_T3': ('T3', 'orange'),
            'KETINGGIAN_T4': ('T4', 'brown')
        }

        output_folder = "WEB_KETINGGIAN_PER_100M"
        os.makedirs(output_folder, exist_ok=True)
        
        # SELALU URUTKAN KM DARI KECIL KE BESAR UNTUK HALAMAN PDF
        unique_km_order = sorted(df['LOKASI_KM'].drop_duplicates().tolist())
        
        output_pdf = f"{output_folder}/{TABLE_NAME}_{LINTAS_NAME}_{TRIP_NAME}_ketinggian_per100m_stasiun.pdf"

        with PdfPages(output_pdf) as pdf:
            for km in unique_km_order:
                chunk = df[df['LOKASI_KM'] == km].copy()
                chunk = chunk[(chunk['LOKASI_M'] >= 0) &
                              (chunk['LOKASI_M'] <= 1000)]
                if chunk.empty:
                    continue

                fig, axs = plt.subplots(
                    nrows=1, ncols=4, figsize=(11.69, 8.27), sharey=True)
                unique_tw = sorted(chunk['TW'].dropna().unique())
                color_list = ['blue', 'red', 'green', 'orange', 'purple']
                tw_colors = {tw: color_list[i % len(
                    color_list)] for i, tw in enumerate(unique_tw)}

                for idx, col in enumerate(available_columns):
                    label, _ = column_mapping[col]
                    ax = axs[idx]

                    for tw in unique_tw:
                        color = tw_colors.get(tw, 'gray')
                        data_tw = chunk[chunk['TW'] == tw]
                        valid_data = data_tw[['LOKASI_M', col]].dropna().copy()
                        if valid_data.empty:
                            continue

                        # Gunakan LOKASI_M langsung tanpa transformasi
                        valid_data['PLOT_Y'] = valid_data['LOKASI_M']
                        valid_data = valid_data.sort_values(by='LOKASI_M')

                        segments = []
                        current_segment = [valid_data.iloc[0]]
                        for j in range(1, len(valid_data)):
                            curr = valid_data.iloc[j]
                            prev = valid_data.iloc[j - 1]
                            if abs(curr['LOKASI_M'] - prev['LOKASI_M']) <= 1:
                                current_segment.append(curr)
                            else:
                                if len(current_segment) > 1:
                                    segments.append(
                                        pd.DataFrame(current_segment))
                                current_segment = [curr]
                        if len(current_segment) > 1:
                            segments.append(pd.DataFrame(current_segment))

                        for seg in segments:
                            ax.plot(seg[col], seg['PLOT_Y'], linestyle='-', linewidth=0.5,
                                    color=color, label=f'TW {tw}' if idx == 0 else None)

                        min_idx = valid_data[col].idxmin()
                        max_idx = valid_data[col].idxmax()
                        min_val = valid_data.loc[min_idx, col]
                        max_val = valid_data.loc[max_idx, col]
                        min_y = valid_data.loc[min_idx, 'PLOT_Y']
                        max_y = valid_data.loc[max_idx, 'PLOT_Y']
                        ax.text(min_val - 1, min_y, f"{min_val:.1f}", fontsize=6,
                                color=color, va='center', ha='right', weight='bold')
                        ax.text(max_val + 1, max_y, f"{max_val:.1f}", fontsize=6,
                                color=color, va='center', ha='left', weight='bold')

                    # Tambahkan garis stasiun
                    for stasiun in stasiun_info:
                        if stasiun['km'] == km:
                            y_pos = stasiun['m']  # Gunakan nilai meter langsung
                            ax.axhline(y=y_pos, color='purple',
                                       linestyle='--', linewidth=1)
                            ax.text(
                                5840, y_pos, stasiun['nama'], fontsize=7, va='center', ha='left', color='purple')

                    ax.axvline(x=4300, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=5300, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=5700, color='gray',
                               linestyle='--', linewidth=0.8)

                    ax.set_title(label, fontsize=10)
                    ax.set_xlim(4000, 5800)

                    # Atur Y axis dari atas 0 ke bawah 1000
                    ax.set_ylim(1000, 0)  # Balik urutan: dari 1000 ke 0
                    y_ticks = range(0, 1001, 100)
                    y_labels = [f"{m}" for m in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels, fontsize=8)

                    ax.grid(True, axis='y', which='major', alpha=0.3)
                    ax.set_xlabel("Ketinggian", fontsize=8)
                    if idx == 0:
                        ax.set_ylabel("LOKASI_M", fontsize=8)

                arah = "Mundur" if is_mundur else "Maju"
                fig.suptitle(
                    f"KM {int(km)} - Ketinggian T1–T4 per 100m per TW | {jalur_info} - {lintas_info} ", 
                    fontsize=12, y=0.96)

                tw_labels = [f'TW {tw}' for tw in unique_tw]
                tw_handles = [plt.Line2D(
                    [0], [0], color=tw_colors[tw], linewidth=2) for tw in unique_tw]
                fig.legend(handles=tw_handles, labels=tw_labels, loc='upper center',
                           bbox_to_anchor=(0.5, 0.92), ncol=len(tw_handles), fontsize=9, frameon=False)

                plt.tight_layout(rect=[0, 0.02, 1, 0.90])
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        return output_pdf


def proses_data_deviasi(TABLE_NAME, LINTAS_NAME, TRIP_NAME, stasiun_info):
    """Fungsi untuk memproses data deviasi jalur"""
    start_km = stasiun_info[0]['km']
    start_m = stasiun_info[0]['m']
    end_km = stasiun_info[-1]['km']
    end_m = stasiun_info[-1]['m']

    # Menggunakan engine yang sudah dikonfigurasi
    engine = get_db_engine()

    with engine.connect() as conn:
        df = pd.read_sql(
            f"""
            SELECT LOKASI_KM, LOKASI_M,
                   DEVIASI_T1, DEVIASI_T2,
                   DEVIASI_T3, DEVIASI_T4,
                   LINTAS, JALUR, id, TW, TRIP
            FROM {TABLE_NAME}
            WHERE LINTAS = '{LINTAS_NAME}' AND TRIP = '{TRIP_NAME}'
            ORDER BY id ASC
            """,
            con=conn
        )

        df = df.dropna(subset=['LOKASI_KM', 'LOKASI_M'])
        df['LOKASI_M'] = df['LOKASI_M'].astype(str).str.replace(',', '.')
        df['LOKASI_M'] = pd.to_numeric(df['LOKASI_M'], errors='coerce')
        df = df.dropna(subset=['LOKASI_M'])
        df['TW'] = df['TW'].astype(str)
        df = df.reset_index(drop=True)

        # Ambil informasi JALUR dan LINTAS dari data pertama
        jalur_info = df['JALUR'].iloc[0] if not df.empty else ''
        lintas_info = df['LINTAS'].iloc[0] if not df.empty else LINTAS_NAME

        # === DEBUG ARAH INPUT ===
        is_mundur_input = (start_km > end_km) or (
            start_km == end_km and start_m > end_m)

        # === FILTER SESUAI RANGE ===
        if is_mundur_input:
            df_filtered = df[
                ((df['LOKASI_KM'] < start_km) & (df['LOKASI_KM'] > end_km)) |
                ((df['LOKASI_KM'] == start_km) & (df['LOKASI_M'] <= start_m)) |
                ((df['LOKASI_KM'] == end_km) & (df['LOKASI_M'] >= end_m))
            ]
        else:
            df_filtered = df[
                ((df['LOKASI_KM'] > start_km) & (df['LOKASI_KM'] < end_km)) |
                ((df['LOKASI_KM'] == start_km) & (df['LOKASI_M'] >= start_m)) |
                ((df['LOKASI_KM'] == end_km) & (df['LOKASI_M'] <= end_m))
            ]

        if df_filtered.empty:
            raise ValueError("Tidak ada data dalam range KM-M yang dipilih.")

        df = df_filtered.copy()

        # === SORT DATA SESUAI ARAH ===
        df = df.sort_values(
            by=['LOKASI_KM', 'LOKASI_M'],
            ascending=[not is_mundur_input, not is_mundur_input]
        ).reset_index(drop=True)

        first_km_in_data = df.iloc[0]['LOKASI_KM']
        last_km_in_data = df.iloc[-1]['LOKASI_KM']
        is_mundur = first_km_in_data > last_km_in_data

        available_columns = ['DEVIASI_T1',
                             'DEVIASI_T2', 'DEVIASI_T3', 'DEVIASI_T4']
        column_mapping = {
            'DEVIASI_T1': ('T1', 'blue'),
            'DEVIASI_T2': ('T2', 'green'),
            'DEVIASI_T3': ('T3', 'orange'),
            'DEVIASI_T4': ('T4', 'brown')
        }

        output_folder = "WEB_DEVIASI_PER_100M"
        os.makedirs(output_folder, exist_ok=True)
        
        # SELALU URUTKAN KM DARI KECIL KE BESAR UNTUK HALAMAN PDF
        unique_km_order = sorted(df['LOKASI_KM'].drop_duplicates().tolist())
        
        output_pdf = f"{output_folder}/{TABLE_NAME}_{LINTAS_NAME}_{TRIP_NAME}_deviasi_per100m_stasiun.pdf"

        with PdfPages(output_pdf) as pdf:
            for km in unique_km_order:
                chunk = df[df['LOKASI_KM'] == km].copy()
                chunk = chunk[(chunk['LOKASI_M'] >= 0) &
                              (chunk['LOKASI_M'] <= 1000)]
                if chunk.empty:
                    continue

                fig, axs = plt.subplots(
                    nrows=1, ncols=4, figsize=(11.69, 8.27), sharey=True)
                unique_tw = sorted(chunk['TW'].dropna().unique())
                color_list = ['blue', 'red', 'green', 'orange', 'purple']
                tw_colors = {tw: color_list[i % len(
                    color_list)] for i, tw in enumerate(unique_tw)}

                for idx, col in enumerate(available_columns):
                    label, _ = column_mapping[col]
                    ax = axs[idx]

                    for tw in unique_tw:
                        color = tw_colors.get(tw, 'gray')
                        data_tw = chunk[chunk['TW'] == tw]
                        valid_data = data_tw[['LOKASI_M', col]].dropna().copy()
                        if valid_data.empty:
                            continue

                        # Gunakan LOKASI_M langsung tanpa transformasi
                        valid_data['PLOT_Y'] = valid_data['LOKASI_M']
                        valid_data = valid_data.sort_values(by='LOKASI_M')

                        segments = []
                        current_segment = [valid_data.iloc[0]]
                        for j in range(1, len(valid_data)):
                            curr = valid_data.iloc[j]
                            prev = valid_data.iloc[j - 1]
                            if abs(curr['LOKASI_M'] - prev['LOKASI_M']) <= 1:
                                current_segment.append(curr)
                            else:
                                if len(current_segment) > 1:
                                    segments.append(
                                        pd.DataFrame(current_segment))
                                current_segment = [curr]
                        if len(current_segment) > 1:
                            segments.append(pd.DataFrame(current_segment))

                        for seg in segments:
                            ax.plot(seg[col], seg['PLOT_Y'], linestyle='-', linewidth=0.5,
                                    color=color, label=f'TW {tw}' if idx == 0 else None)

                        min_idx = valid_data[col].idxmin()
                        max_idx = valid_data[col].idxmax()
                        min_val = valid_data.loc[min_idx, col]
                        max_val = valid_data.loc[max_idx, col]
                        min_y = valid_data.loc[min_idx, 'PLOT_Y']
                        max_y = valid_data.loc[max_idx, 'PLOT_Y']
                        ax.text(min_val - 1, min_y, f"{min_val:.1f}", fontsize=6,
                                color=color, va='center', ha='right', weight='bold')
                        ax.text(max_val + 1, max_y, f"{max_val:.1f}", fontsize=6,
                                color=color, va='center', ha='left', weight='bold')

                    # Tambahkan garis stasiun
                    for stasiun in stasiun_info:
                        if stasiun['km'] == km:
                            y_pos = stasiun['m']  # Gunakan nilai meter langsung
                            ax.axhline(y=y_pos, color='purple',
                                       linestyle='--', linewidth=1)
                            ax.text(
                                620, y_pos, stasiun['nama'], fontsize=7, va='center', ha='left', color='purple')

                    ax.axvline(x=-300, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=-200, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=200, color='gray',
                               linestyle='--', linewidth=0.8)
                    ax.axvline(x=300, color='gray',
                               linestyle='--', linewidth=0.8)

                    ax.set_title(label, fontsize=10)
                    ax.set_xlim(-600, 600)

                    # SELALU GUNAKAN URUTAN 0-1000, TANPA MEMBALIK
                    ax.set_ylim(0, 1000)
                    y_ticks = range(0, 1001, 100)
                    y_labels = [f"{m}" for m in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels, fontsize=8)

                    ax.grid(True, axis='y', which='major', alpha=0.3)
                    ax.set_xlabel("Deviasi", fontsize=8)
                    if idx == 0:
                        ax.set_ylabel("LOKASI_M", fontsize=8)

                arah = "Mundur" if is_mundur else "Maju"
                fig.suptitle(
                    f"KM {int(km)} - Deviasi T1–T4 per 100m per TW | {jalur_info} - {lintas_info} ", 
                    fontsize=12, y=0.96)

                tw_labels = [f'TW {tw}' for tw in unique_tw]
                tw_handles = [plt.Line2D(
                    [0], [0], color=tw_colors[tw], linewidth=2) for tw in unique_tw]
                fig.legend(handles=tw_handles, labels=tw_labels, loc='upper center',
                           bbox_to_anchor=(0.5, 0.92), ncol=len(tw_handles), fontsize=9, frameon=False)

                plt.tight_layout(rect=[0, 0.02, 1, 0.90])
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        return output_pdf


# Untuk backward compatibility, tetap ada fungsi proses_data yang redirect ke ketebalan
def proses_data(TABLE_NAME, LINTAS_NAME, TRIP_NAME, stasiun_info):
    """Fungsi backward compatibility - redirect ke ketebalan"""
    return proses_data_ketebalan(TABLE_NAME, LINTAS_NAME, TRIP_NAME, stasiun_info)


@app.route('/pdf/<filename>')
def serve_pdf(filename):
    # Deteksi folder berdasarkan nama file
    if 'ketebalan' in filename.lower():
        folder = 'WEB_KETEBALAN_PER_100M'
    elif 'ketinggian' in filename.lower():
        folder = 'WEB_KETINGGIAN_PER_100M'
    elif 'deviasi' in filename.lower():
        folder = 'WEB_DEVIASI_PER_100M'
    else:
        # Default ke ketebalan untuk backward compatibility
        folder = 'WEB_KETEBALAN_PER_100M'

    return send_from_directory(folder, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
