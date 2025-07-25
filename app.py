import streamlit as st
import os
from transformers import pipeline
import gspread
from datetime import datetime

# --- Configuration ---
MODEL_PATH = "./emosic_emotion_classifier_model"
GOOGLE_SHEET_NAME = "EmoSic_Feedback"

# --- 1. Load the Emotion Classification Model ---
@st.cache_resource
def load_emotion_model(model_directory):
    """Loads the pre-trained emotion classification pipeline from the specified directory."""
    try:
        classifier = pipeline("sentiment-analysis", model=model_directory, tokenizer=model_directory)
        return classifier
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        st.info("Please ensure the 'emosic_emotion_classifier_model' folder is in the same directory as app.py "
                "and contains all model files (pytorch_model.bin/model.safetensors, config.json, tokenizer.json, etc.).")
        st.stop()

emotion_classifier = load_emotion_model(MODEL_PATH)


# --- 2. Define Song Playlists (UPDATED FOR ENGLISH YOUTUBE LINKS) ---
EMOTION_PLAYLISTS = {
    "joy": {
        "English": [
            {"title": "Happy - Pharrell Williams", "url": "https://youtu.be/ZbZSe6N_BXs?si=jsNxUVwIUFrHacsZ"},
            {"title": "Ed Sheeran - Sapphire", "url": "https://youtu.be/JgDNFQ2RaLQ?si=qDSygwPlr5mb-NkZ"},
            {"title": "Walking on Sunshine - Katrina and the Waves", "url": "https://www.youtube.com/results?search_query=Katrina+and+the+Waves+Walking+on+Sunshine"},
            {"title": "Can't Stop the Feeling! - Justin Timberlake", "url": "https://www.youtube.com/results?search_query=Justin+Timberlake+Can%27t+Stop+the+Feeling"},
            {"title": "Shake It Off - Taylor Swift", "url": "https://www.youtube.com/results?search_query=Taylor+Swift+Shake+It+Off"},
            {"title": "Treasure - Bruno Mars", "url": "https://www.youtube.com/results?search_query=Bruno+Mars+Treasure"},
            {"title": "Levitating - Dua Lipa", "url": "https://www.youtube.com/results?search_query=Dua+Lipa+Levitating"},
            {"title": "Uptown Funk - Mark Ronson ft. Bruno Mars", "url": "https://www.youtube.com/results?search_query=Mark+Ronson+Bruno+Mars+Uptown+Funk"},
            {"title": "I Gotta Feeling - Black Eyed Peas", "url": "https://www.youtube.com/results?search_query=Black+Eyed+Peas+I+Gotta+Feeling"},
            {"title": "Good Life - OneRepublic", "url": "https://www.youtube.com/results?search_query=OneRepublic+Good+Life"},
            {"title": "A Sky Full of Stars - Coldplay", "url": "https://www.youtube.com/results?search_query=Coldplay+A+Sky+Full+of+Stars"}
        ],
        "Hindi": [
            {"title": "Dil Dhadakne Do", "url": "https://www.youtube.com/results?search_query=Dil+Dhadakne+Do+song"},
            {"title": "Tareefan", "url": "https://www.youtube.com/results?search_query=Tareefan+song"},
            {"title": "Gallan Goodiyaan", "url": "https://www.youtube.com/results?search_query=Gallan+Goodiyaan+song"},
            {"title": "Jai Jai Shivshankar", "url": "https://www.youtube.com/results?search_query=Jai+Jai+Shivshankar+song"},
            {"title": "Badtameez Dil", "url": "https://www.youtube.com/results?search_query=Badtameez+Dil+song"}
        ],
        "Telugu": [
            {"title": "Butta Bomma", "url": "https://www.youtube.com/results?search_query=Butta+Bomma+song"},
            {"title": "Ramuloo Ramulaa", "url": "https://www.youtube.com/results?search_query=Ramuloo+Ramulaa+song"},
            {"title": "Samajavaragamana", "url": "https://www.youtube.com/results?search_query=Samajavaragamana+song"},
            {"title": "Sarrainodu", "url": "https://www.youtube.com/results?search_query=Sarrainodu+song"},
            {"title": "Nee Jathaga", "url": "https://www.youtube.com/results?search_query=Nee+Jathaga+song"}
        ],
        "Kannada": [
            {"title": "Belageddu", "url": "https://www.youtube.com/results?search_query=Belageddu+song"},
            {"title": "Neene Neene", "url": "https://www.youtube.com/results?search_query=Neene+Neene+song"},
            {"title": "Salaam Rocky Bhai", "url": "https://www.youtube.com/results?search_query=Salaam+Rocky+Bhai+song"},
            {"title": "Yajamana", "url": "https://www.youtube.com/results?search_query=Yajamana+song"},
            {"title": "Geleya Nanna Geleya", "url": "https://www.youtube.com/results?search_query=Geleya+Nanna+Geleya+song"}
        ]
    },
    "sadness": {
        "English": [
            {"title": "Someone Like You - Adele", "url": "https://www.youtube.com/results?search_query=Adele+Someone+Like+You"},
            {"title": "Too Good at Goodbyes - Sam Smith", "url": "https://www.youtube.com/results?search_query=Sam+Smith+Too+Good+at+Goodbyes"},
            {"title": "Everything I Wanted - Billie Eilish", "url": "https://www.youtube.com/results?search_query=Billie+Eilish+Everything+I+Wanted"},
            {"title": "Someone You Loved - Lewis Capaldi", "url": "https://www.youtube.com/results?search_query=Lewis+Capaldi+Someone+You+Loved"},
            {"title": "Goodbye My Lover - James Blunt", "url": "https://www.youtube.com/results?search_query=James+Blunt+Goodbye+My+Lover"},
            {"title": "Let Her Go - Passenger", "url": "https://www.youtube.com/results?search_query=Passenger+Let+Her+Go"},
            {"title": "Fix You - Coldplay", "url": "https://www.youtube.com/results?search_query=Coldplay+Fix+You"},
            {"title": "Drivers License - Olivia Rodrigo", "url": "https://www.youtube.com/results?search_query=Olivia+Rodrigo+Drivers+License"},
            {"title": "Photograph - Ed Sheeran", "url": "https://www.youtube.com/results?search_query=Ed+Sheeran+Photograph"},
            {"title": "Breathe Me - Sia", "url": "https://www.youtube.com/results?search_query=Sia+Breathe+Me"}
        ],
        "Hindi": [
            {"title": "Kabira", "url": "https://www.youtube.com/results?search_query=Kabira+song"},
            {"title": "Channa Mereya", "url": "https://www.youtube.com/results?search_query=Channa+Mereya+song"},
            {"title": "Tujhe Kitna Chahne Lage", "url": "https://www.youtube.com/results?search_query=Tujhe+Kitna+Chahne+Lage+song"},
            {"title": "Agar Tum Saath Ho", "url": "https://www.youtube.com/results?search_query=Agar+Tum+Saath+Ho+song"},
            {"title": "Main Yahaan Hoon", "url": "https://www.youtube.com/results?search_query=Main+Yahaan+Hoon+song"}
        ],
        "Telugu": [
            {"title": "Nuvve Nuvve", "url": "https://www.youtube.com/results?search_query=Nuvve+Nuvve+song"},
            {"title": "Yeduta Nilichindi", "url": "https://www.youtube.com/results?search_query=Yeduta+Nilichindi+song"},
            {"title": "Nee Kallalona", "url": "https://www.youtube.com/results?search_query=Nee+Kallalona+song"},
            {"title": "Oohalu Gusagusalade", "url": "https://www.youtube.com/results?search_query=Oohalu+Gusagusalade+song"},
            {"title": "Mounamgaane", "url": "https://www.youtube.com/results?search_query=Mounamgaane+song"}
        ],
        "Kannada": [
            {"title": "Ninnindale", "url": "https://www.youtube.com/results?search_query=Ninnindale+song"},
            {"title": "Ee Sanje", "url": "https://www.youtube.com/results?search_query=Ee+Sanje+song"},
            {"title": "Kanasalu Neene", "url": "https://www.youtube.com/results?search_query=Kanasalu+Neene+song"},
            {"title": "Neeralli Sanna", "url": "https://www.youtube.com/results?search_query=Neeralli+Sanna+song"},
            {"title": "Usire Usire", "url": "https://www.youtube.com/results?search_query=Usire+Usire+song"}
        ]
    },
    "anger": {
        "English": [
            {"title": "Numb - Linkin Park", "url": "https://www.youtube.com/results?search_query=Linkin+Park+Numb"},
            {"title": "Lose Yourself - Eminem", "url": "https://www.youtube.com/results?search_query=Eminem+Lose+Yourself"},
            {"title": "Believer - Imagine Dragons", "url": "https://www.youtube.com/results?search_query=Imagine+Dragons+Believer"},
            {"title": "Boulevard of Broken Dreams - Green Day", "url": "https://www.youtube.com/results?search_query=Green+Day+Boulevard+of+Broken+Dreams"},
            {"title": "Smells Like Teen Spirit - Nirvana", "url": "https://www.youtube.com/results?search_query=Nirvana+Smells+Like+Teen+Spirit"},
            {"title": "Disturbia - Rihanna", "url": "https://www.youtube.com/results?search_query=Rihanna+Disturbia"},
            {"title": "You Should See Me in a Crown - Billie Eilish", "url": "https://www.youtube.com/results?search_query=Billie+Eilish+You+Should+See+Me+in+a+Crown"},
            {"title": "Bring Me to Life - Evanescence", "url": "https://www.youtube.com/results?search_query=Evanescence+Bring+Me+to+Life"},
            {"title": "Sorry Not Sorry - Demi Lovato", "url": "https://www.youtube.com/results?search_query=Demi+Lovato+Sorry+Not+Sorry"},
            {"title": "I'm Not Okay (I Promise) - My Chemical Romance", "url": "https://www.youtube.com/results?search_query=My+Chemical+Romance+I%27m+Not+Okay"}
        ],
        "Hindi": [
            {"title": "Ranjha Ranjha", "url": "https://www.youtube.com/results?search_query=Ranjha+Ranjha+song"},
            {"title": "Bekhayali", "url": "https://www.youtube.com/results?search_query=Bekhayali+song"},
            {"title": "Gali Gali", "url": "https://www.youtube.com/results?search_query=Gali+Gali+song"},
            {"title": "Zinda", "url": "https://www.youtube.com/results?search_query=Zinda+song"},
            {"title": "Kaala Chashma (from Baar Baar Dekho)", "url": "https://www.youtube.com/results?search_query=Kaala+Chashma+song"}
        ],
        "Telugu": [
            {"title": "Bad Boy - Businessman", "url": "https://www.youtube.com/results?search_query=Bad+Boy+Businessman+song"},
            {"title": "Top Lesi Poddi - Iddarammayilatho", "url": "https://www.youtube.com/results?search_query=Top+Lesi+Poddi+Iddarammayilatho+song"},
            {"title": "Devudaa Devudaa - Gabbar Singh", "url": "https://www.youtube.com/results?search_query=Devudaa+Devudaa+Gabbar+Singh+song"},
            {"title": "Blockbuster - Sarrainodu", "url": "https://www.youtube.com/results?search_query=Blockbuster+Sarrainodu+song"}
        ],
        "Kannada": [
            {"title": "KGF Theme Song", "url": "https://www.youtube.com/results?search_query=KGF+Theme+Song"},
            {"title": "Tagaru Banthu Tagaru", "url": "https://www.youtube.com/results?search_query=Tagaru+Banthu+Tagaru+song"},
            {"title": "Chuttu Chuttu - Raambo 2", "url": "https://www.youtube.com/results?search_query=Chuttu+Chuttu+Raambo+2+song"},
            {"title": "Dheera Dheera - K.G.F: Chapter 1", "url": "https://www.youtube.com/results?search_query=Dheera+Dheera+K.G.F+Chapter+1+song"}
        ]
    },
    "fear": {
        "English": [
            {"title": "Creep - Radiohead", "url": "https://www.youtube.com/results?search_query=Radiohead+Creep"},
            {"title": "Jealous - Labrinth", "url": "https://www.youtube.com/results?search_query=Labrinth+Jealous"},
            {"title": "Elastic Heart - Sia", "url": "https://www.youtube.com/results?search_query=Sia+Elastic+Heart"},
            {"title": "Take Me to Church - Hozier", "url": "https://www.youtube.com/results?search_query=Hozier+Take+Me+to+Church"},
            {"title": "Demons - Imagine Dragons", "url": "https://www.youtube.com/results?search_query=Imagine+Dragons+Demons"},
            {"title": "Runaway - Aurora", "url": "https://www.youtube.com/results?search_query=Aurora+Runaway"},
            {"title": "Shake It Out - Florence + The Machine", "url": "https://www.youtube.com/results?search_query=Florence+The+Machine+Shake+It+Out"},
            {"title": "When the Party’s Over - Billie Eilish", "url": "https://www.youtube.com/results?search_query=Billie+Eilish+When+the+Party%E2%80%99s+Over"},
            {"title": "Liability - Lorde", "url": "https://www.youtube.com/results?search_query=Lorde+Liability"},
            {"title": "Somewhere Only We Know - Keane", "url": "https://www.youtube.com/results?search_query=Keane+Somewhere+Only+We+Know"}
        ],
        "Hindi": [
            {"title": "Aashayein", "url": "https://www.youtube.com/results?search_query=Aashayein+song"},
            {"title": "Bandheya", "url": "https://www.youtube.com/results?search_query=Bandheya+song"},
            {"title": "Zinda Hoon Yaar", "url": "https://www.youtube.com/results?search_query=Zinda+Hoon+Yaar+song"},
            {"title": "Tu Hi Meri Shab Hai", "url": "https://www.youtube.com/results?search_query=Tu+Hi+Meri+Shab+Hai+song"},
            {"title": "Darr (Title Song)", "url": "https://www.youtube.com/results?search_query=Darr+Title+Song"}
        ],
        "Telugu": [
            {"title": "Gundello Emundo - Pelli Sandadi", "url": "https://www.youtube.com/results?search_query=Gundello+Emundo+Pelli+Sandadi+song"},
            {"title": "Nijame Ne Chebutunna - Kothabangarulokam", "url": "https://www.youtube.com/results?search_query=Nijame+Ne+Chebutunna+Kothabangarulokam+song"},
            {"title": "Oka Laila Kosam - Oka Laila Kosam", "url": "https://www.youtube.com/results?search_query=Oka+Laila+Kosam+song"},
            {"title": "Manase Manase - Vasantham", "url": "https://www.youtube.com/results?search_query=Manase+Manase+Vasantham+song"}
        ],
        "Kannada": [
            {"title": "Gooli - Gooli", "url": "https://www.youtube.com/results?search_query=Gooli+song"},
            {"title": "Jotheyali - Ninnindale", "url": "https://www.youtube.com/results?search_query=Jotheyali+Ninnindale+song"},
            {"title": "Nee Bandu Nintaaga - Gajakesari", "url": "https://www.youtube.com/results?search_query=Nee+Bandu+Nintaaga+Gajakesari+song"},
            {"title": "Kariya I Love You - Kariya", "url": "https://www.youtube.com/results?search_query=Kariya+I+Love+You+song"}
        ]
    },
    "love": {
        "English": [
            {"title": "Perfect - Ed Sheeran", "url": "https://www.youtube.com/results?search_query=Ed+Sheeran+Perfect"},
            {"title": "All of Me - John Legend", "url": "https://www.youtube.com/results?search_query=John+Legend+All+of+Me"},
            {"title": "Lover - Taylor Swift", "url": "https://www.youtube.com/results?search_query=Taylor+Swift+Lover"},
            {"title": "Just the Way You Are - Bruno Mars", "url": "https://www.youtube.com/results?search_query=Bruno+Mars+Just+the+Way+You+Are"},
            {"title": "Love Me Like You Do - Ellie Goulding", "url": "https://www.youtube.com/results?search_query=Ellie+Goulding+Love+Me+Like+You+Do"},
            {"title": "Say You Won’t Let Go - James Arthur", "url": "https://www.youtube.com/results?search_query=James+Arthur+Say+You+Won%E2%80%99t+Let+Go"},
            {"title": "Earned It - The Weeknd", "url": "https://www.youtube.com/results?search_query=The+Weeknd+Earned+It"},
            {"title": "Young and Beautiful - Lana Del Rey", "url": "https://www.youtube.com/results?search_query=Lana+Del+Rey+Young+and+Beautiful"},
            {"title": "A Thousand Years - Christina Perri", "url": "https://www.youtube.com/results?search_query=Christina+Perri+A+Thousand+Years"},
            {"title": "Halo - Beyoncé", "url": "https://www.youtube.com/results?search_query=Beyonc%C3%A9+Halo"}
        ],
        "Hindi": [
            {"title": "Tere Mere Sapne", "url": "https://www.youtube.com/results?search_query=Tere+Mere+Sapne+song"},
            {"title": "Raabta", "url": "https://www.youtube.com/results?search_query=Raabta+song"},
            {"title": "Tum Hi Ho", "url": "https://www.youtube.com/results?search_query=Tum+Hi+Ho+song"},
            {"title": "Dil Diyan Gallan", "url": "https://www.youtube.com/results?search_query=Dil+Diyan+Gallan+song"},
            {"title": "Pehla Nasha", "url": "https://www.youtube.com/results?search_query=Pehla+Nasha+song"}
        ],
        "Telugu": [
            {"title": "Nee Choopule - Endukante Premanta", "url": "https://www.youtube.com/results?search_query=Nee+Choopule+Endukante+Premanta+song"},
            {"title": "Yeto Vellipoyindhi Manasu - Yeto Vellipoyindhi Manasu", "url": "https://www.youtube.com/results?search_query=Yeto+Vellipoyindhi+Manasu+song"},
            {"title": "Ninnu Kori - Ninnu Kori", "url": "https://www.youtube.com/results?search_query=Ninnu+Kori+song"},
            {"title": "Priyatama Priyatama - Majili", "url": "https://www.youtube.com/results?search_query=Priyatama+Priyatama+Majili+song"}
        ],
        "Kannada": [
            {"title": "Preethi Maayavi - Mungaru Male", "url": "https://www.youtube.com/results?search_query=Preethi+Maayavi+Mungaru+Male+song"},
            {"title": "Neene Neene - Akash", "url": "https://www.youtube.com/results?search_query=Neene+Neene+Akash+song"},
            {"title": "Usire Usire - Huccha", "url": "https://www.youtube.com/results?search_query=Usire+Usire+Huccha+song"},
            {"title": "Nee Sanihake - Chakravyuha", "url": "https://www.youtube.com/results?search_query=Nee+Sanihake+Chakravyuha+song"}
        ]
    },
    "surprise": {
        "English": [
            {"title": "Paradise - Coldplay", "url": "https://www.youtube.com/results?search_query=Coldplay+Paradise"},
            {"title": "Fireflies - Owl City", "url": "https://www.youtube.com/results?search_query=Owl+City+Fireflies"},
            {"title": "Counting Stars - OneRepublic", "url": "https://www.youtube.com/results?search_query=OneRepublic+Counting+Stars"},
            {"title": "On Top of the World - Imagine Dragons", "url": "https://www.youtube.com/results?search_query=Imagine+Dragons+On+Top+of+the+World"},
            {"title": "Ocean Eyes - Billie Eilish", "url": "https://www.youtube.com/results?search_query=Billie+Eilish+Ocean+Eyes"},
            {"title": "Cosmic Love - Florence + The Machine", "url": "https://www.youtube.com/results?search_query=Florence+The+Machine+Cosmic+Love"},
            {"title": "Midnight City - M83", "url": "https://www.youtube.com/results?search_query=M83+Midnight+City"},
            {"title": "Castle on the Hill - Ed Sheeran", "url": "https://www.youtube.com/results?search_query=Ed+Sheeran+Castle+on+the+Hill"},
            {"title": "Wish You Were Here - Pink Floyd", "url": "https://www.youtube.com/results?search_query=Pink+Floyd+Wish+You+Were+Here"},
            {"title": "Youth - Troye Sivan", "url": "https://www.youtube.com/results?search_query=Troye+Sivan+Youth"}
        ],
        "Hindi": [
            {"title": "Chaiyya Chaiyya", "url": "https://www.youtube.com/results?search_query=Chaiyya+Chaiyya+song"},
            {"title": "Kala Chashma", "url": "https://www.youtube.com/results?search_query=Kala+Chashma+song"},
            {"title": "London Thumakda", "url": "https://www.youtube.com/results?search_query=London+Thumakda+song"},
            {"title": "Balam Pichkari", "url": "https://www.youtube.com/results?search_query=Balam+Pichkari+song"},
            {"title": "Ghungroo", "url": "https://www.youtube.com/results?search_query=Ghungroo+song"}
        ],
        "Telugu": [
            {"title": "Ringa Ringa - Arya 2", "url": "https://www.youtube.com/results?search_query=Ringa+Ringa+Arya+2+song"},
            {"title": "Pakka Local - Janatha Garage", "url": "https://www.youtube.com/results?search_query=Pakka+Local+Janatha+Garage+song"},
            {"title": "Ammadu Let's Do Kummudu - Khaidi No. 150", "url": "https://www.youtube.com/results?search_query=Ammadu+Let%27s+Do+Kummudu+Khaidi+No.+150+song"},
            {"title": "Dimaak Kharaab - Ismart Shankar", "url": "https://www.youtube.com/results?search_query=Dimaak+Kharaab+Ismart+शंकर+song"}
        ],
        "Kannada": [
            {"title": "Bombat - Raambo 2", "url": "https://www.youtube.com/results?search_query=Bombat+Raambo+2+song"},
            {"title": "Karabuu - Pogaru", "url": "https://www.youtube.com/results?search_query=Karabuu+Pogaru+song"},
            {"title": "Dostha Kano - Chakravyuha", "url": "https://www.youtube.com/results?search_query=Dostha+Kano+Chakravyuha+song"},
            {"title": "Appu Dance - Appu", "url": "https://www.youtube.com/results?search_query=Appu+Dance+Appu+song"}
        ]
    }
}

# --- 3. Google Sheets Integration Functions ---
@st.cache_resource
def get_google_sheet_client():
    """Authenticates with Google Sheets API using Streamlit secrets and returns a gspread client."""
    try:
        creds = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
        }
        gc = gspread.service_account_from_dict(creds)
        return gc
    except KeyError:
        st.error("Google Sheets API key not found in Streamlit secrets. "
                 "Please ensure 'gcp_service_account' is configured in your app's secrets.")
        st.stop()
    except Exception as e:
        st.error(f"Error authenticating with Google Sheets: {e}. "
                 "Please check your Google Cloud setup and `st.secrets` configuration.")
        st.stop()

def log_feedback_to_sheet(user_input, detected_emotion, language, accuracy_feedback, comment_feedback):
    """Logs user feedback to the specified Google Sheet."""
    try:
        client = get_google_sheet_client()
        spreadsheet = client.open(GOOGLE_SHEET_NAME)
        worksheet = spreadsheet.sheet1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_row = [timestamp, user_input, detected_emotion, language, accuracy_feedback, comment_feedback]
        worksheet.append_row(data_row)
        st.success("Feedback submitted successfully! Thank you for helping us improve.")
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Google Sheet '{GOOGLE_SHEET_NAME}' not found. Please ensure the sheet exists and its name matches.")
    except Exception as e:
        st.error(f"Error logging feedback to Google Sheet: {e}. "
                 "Ensure the service account has 'Editor' access to the sheet.")


# --- 4. Streamlit UI Design ---
st.set_page_config(
    page_title="EmoSic 🎵",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 EmoSic")
st.caption("✨ Every Feeling Deserves a Song ✨")

st.write(
    "Tell us how you feel — EmoSic will detect your emotion and suggest a perfect playlist "
    "in your favorite language! 🎵"
)

st.divider()

# --- Sidebar for Important Notes ---
st.sidebar.header("Important Notes")
st.sidebar.info("""
    **Emotion Detection Model:**
    The model used (DistilRoBERTa) is primarily trained on **English** text.
    Accuracy for non-English input may be limited.

    **Music Recommendations:**
    The language selector primarily affects the language of song recommendations.
""")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ by Shrividya | EmoSic 🎵")


# --- Main Content: Emotion Analysis ---
st.subheader("💭 Tell us how you feel:")
user_input_text = st.text_area(
    "Describe your mood 👇",
    placeholder="e.g. I feel so energetic and happy today! 🌟",
    height=120,
    key="emotion_text_area"
)

if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = None
if 'user_text_for_feedback' not in st.session_state:
    st.session_state.user_text_for_feedback = ""


# Predict emotion when button is clicked
if st.button("🎵 Get My Playlist!", key="get_playlist_button"):
    if user_input_text.strip() == "":
        st.warning("🚫 Please write something to get your playlist!")
        st.session_state.detected_emotion = None
    else:
        with st.spinner("Analyzing emotion... Please wait."):
            prediction = emotion_classifier(user_input_text)
            emotion = prediction[0]['label'].lower()
            confidence_score = prediction[0]['score'] * 100

            st.session_state.detected_emotion = emotion
            st.session_state.user_text_for_feedback = user_input_text

            st.success(f"✅ Detected Emotion: **{emotion.title()}** (Confidence: {confidence_score:.2f}%)")
            st.divider()

# --- 5. Display Song Recommendations ---
if st.session_state.detected_emotion:
    emotion_key = st.session_state.detected_emotion

    st.subheader("🎙️ Pick your preferred language for music:")
    languages_available = list(EMOTION_PLAYLISTS[emotion_key].keys())
    try:
        default_lang_index = languages_available.index(st.session_state.get("music_lang_select", "English"))
    except ValueError:
        default_lang_index = 0

    language_choice = st.selectbox(
        "Choose your language:",
        languages_available,
        index=default_lang_index,
        key="main_lang_choice"
    )

    if emotion_key in EMOTION_PLAYLISTS and language_choice in EMOTION_PLAYLISTS[emotion_key]:
        songs = EMOTION_PLAYLISTS[emotion_key][language_choice]

        if songs:
            st.subheader(f"🎶 Songs for you in {language_choice}:")
            for song in songs:
                # Display song title as a clickable YouTube link
                # Assumes 'song' is a dictionary with 'title' and 'url' keys
                st.markdown(f"🎵 [{song['title']}]({song['url']})")
        else:
            st.warning(f"😢 Sorry! No songs found for {emotion_key.title()} in {language_choice}.")
    else:
        st.info("😢 Sorry! No songs found for this emotion yet. Stay tuned!")

# --- 6. Feedback System ---
st.divider()
st.subheader("🙌 Give Us Your Feedback")
st.markdown("Your thoughts help us grow! Let us know if the emotion detection was accurate and suggest new songs.")

with st.form(key='feedback_form'):
    feedback_emotion_accuracy = st.radio(
        "✨ Was EmoSic helpful?",
        ["😍 Loved it!", "🙂 It was okay", "😕 Needs work"],
        key="emotion_accuracy_radio"
    )
    feedback_comments = st.text_area(
        "💡 Suggestions to make EmoSic better:",
        placeholder="Your thoughts help us grow!",
        key="comments_text_area"
    )
    submit_feedback_button = st.form_submit_button("📬 Submit Feedback")

    if submit_feedback_button:
        log_feedback_to_sheet(
            st.session_state.user_text_for_feedback,
            st.session_state.detected_emotion.title() if st.session_state.detected_emotion else "N/A",
            st.session_state.get("main_lang_choice", "N/A"),
            feedback_emotion_accuracy,
            feedback_comments
        )
