import os
import datetime
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

def get_today():
    return str(datetime.date.today())

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "postgres"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "6677"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def get_all_meetings_for_user(user_id: str):
    """
    Fetches the actual meetings from PostgreSQL where the user is either the host or a participant.
    """
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        try:
            numeric_user_id = int(user_id)
        except ValueError:
            numeric_user_id = user_id

        query = """
            WITH user_meetings AS (
                SELECT DISTINCT m.id
                FROM meetings m
                LEFT JOIN meeting_participants mp ON m.id = mp.meeting_id
                WHERE m.host_id = %s OR mp.user_id = %s
            )
            SELECT 
                m.room_id,
                m.title,
                m.start_timeutc
            FROM meetings m
            JOIN user_meetings um ON m.id = um.id
        """
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, (numeric_user_id, numeric_user_id))
        rows = cursor.fetchall()
        
        meetings = []
        for row in rows:
            dt = row.get("start_timeutc")
            date_str = str(dt.date()) if dt else ""
            time_str = str(dt.time())[:5] if dt else "" # Format HH:MM
            
            meetings.append({
                "room_id": row.get("room_id", ""),
                "title": row.get("title", ""),
                "date": date_str,
                "time": time_str
            })
            
        cursor.close()
        conn.close()
        return meetings
            
    except Exception as e:
        print(f"Error fetching meetings for user {user_id}: {e}")
        if conn:
            conn.close()
        return []
