# Install required packages:
# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import re
import PyPDF2

# Safe import for docx (won't crash if not installed)
try:
    import docx
except ImportError:
    docx = None


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    try:
        svc_model = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        le = pickle.load(open('encoder.pkl', 'rb'))
        return svc_model, tfidf, le
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


svc_model, tfidf, le = load_models()


# ================= TEXT CLEANING =================
def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', ' ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt


# ================= FILE READ FUNCTIONS =================
def extract_text_from_pdf(file):
    text = ''
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    except Exception as e:
        raise Exception(f"PDF read error: {e}")
    return text


def extract_text_from_docx(file):
    if docx is None:
        raise Exception("DOCX support not available. Install using: pip install python-docx")

    try:
        document = docx.Document(file)
        text = "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        raise Exception(f"DOCX read error: {e}")

    return text


def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except:
        return file.read().decode('latin-1')


# ================= FILE HANDLER =================
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)

    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)

    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)

    else:
        raise Exception("Unsupported file type")


# ================= PREDICTION =================
def predict_category(resume_text):
    cleaned = cleanResume(resume_text)
    vector = tfidf.transform([cleaned]).toarray()
    pred = svc_model.predict(vector)
    return le.inverse_transform(pred)[0]


# ================= STREAMLIT UI =================
def main():
    st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")

    st.title("📄 Resume Category Prediction")
    st.write("Upload your resume (PDF, DOCX, TXT) to predict job category")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            text = handle_file_upload(uploaded_file)

            if not text.strip():
                st.warning("No text extracted from file")
                return

            st.success("Text extracted successfully!")

            if st.checkbox("Show Extracted Text"):
                st.text_area("Resume Content", text, height=300)

            if svc_model is None:
                st.error("Model not loaded properly")
                return

            category = predict_category(text)

            st.subheader("🎯 Predicted Category")
            st.success(category)

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()