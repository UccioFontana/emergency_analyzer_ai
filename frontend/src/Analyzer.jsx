import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FaMicrophone, FaMicrophoneSlash, FaArrowRight } from "react-icons/fa";

function Analyzer() {
  const [inputMode, setInputMode] = useState(null); // "text" | "voice" | null
  const [text, setText] = useState("");
  const [response, setResponse] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [partialText, setPartialText] = useState("");
  const [loading, setLoading] = useState(false);
  const [recognition, setRecognition] = useState(null);

  // ✅ Configura SpeechRecognition
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recog = new SpeechRecognition();
      recog.lang = "it-IT";
      recog.interimResults = true;
      recog.continuous = false;

      recog.onresult = (event) => {
        let transcript = "";
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setPartialText(transcript);
        if (event.results[0].isFinal) {
          setIsRecording(false);
          analyzeVoice(transcript);
        }
      };

      recog.onerror = (event) => {
        console.error("Errore nel riconoscimento vocale:", event.error);
        setIsRecording(false);
        setResponse("Errore durante la registrazione.");
      };

      setRecognition(recog);
    }
  }, []);

  // ✅ Analisi testo
  const handleTextSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResponse("Analisi in corso... 🔍");
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ testo: text }),
      });
      const data = await res.json();
      setResponse(formatResponse(data));
    } catch (error) {
      console.error(error);
      setResponse("Errore durante l'analisi del testo.");
    }
    setLoading(false);
    setText("");
  };

  // ✅ Analisi voce (dopo trascrizione)
  const analyzeVoice = async (transcript) => {
    setLoading(true);
    setResponse("Analisi della registrazione in corso... 🔍");
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ testo: transcript }),
      });
      const data = await res.json();
      setResponse(formatResponse(data));
    } catch (error) {
      console.error(error);
      setResponse("Errore durante l'analisi vocale.");
    }
    setLoading(false);
  };

  // ✅ Avvia/ferma registrazione vocale
  const handleVoiceToggle = () => {
    if (!recognition) {
      alert("Il riconoscimento vocale non è supportato su questo browser.");
      return;
    }
    if (isRecording) {
      recognition.stop();
      setIsRecording(false);
    } else {
      setInputMode("voice");
      setIsRecording(true);
      setResponse("");
      setPartialText("");
      recognition.start();
    }
  };

  // ✅ Formatta la risposta in modo leggibile
  const formatResponse = (data) => {
    return `Gravità: ${data.gravita_predetta}\nTipologia: ${data.tipologia_predetta}\nEnti: ${data.enti_predetti}`;
  };

  return (
    <div className="bg-gray-50 text-gray-900 min-h-screen font-sans overflow-x-hidden">
      {/* HEADER */}
      <header className="fixed top-0 left-0 w-full z-50 bg-gradient-to-r from-red-700 to-red-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-extrabold uppercase">
            Vigili del Fuoco <span className="text-yellow-400">Matera</span>
          </h1>
          <nav className="flex gap-8 text-lg font-semibold">
            <a href="/" className="hover:text-yellow-400 transition">Home</a>
            <a href="#contatti" className="hover:text-yellow-400 transition">Contatti</a>
          </nav>
        </div>
      </header>

      {/* HERO */}
      <section className="relative h-[50vh] flex justify-center items-center text-center bg-gradient-to-br from-red-800 to-red-600 text-white pt-24">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="z-10"
        >
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 drop-shadow-lg">
            Analisi Emergenze
          </h1>
          <p className="text-xl text-gray-100">
            Inserisci il testo oppure usa la tua voce per avviare l’analisi.
          </p>
        </motion.div>
      </section>

      {/* SEZIONE ANALIZZATORE */}
      <section className="max-w-6xl mx-auto px-6 py-16 grid grid-cols-1 md:grid-cols-2 gap-12">
        {/* Card Input Testo */}
        <motion.div
          initial={{ opacity: 0, x: -80 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className={`backdrop-blur-md bg-white/30 p-8 rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition ${
            inputMode === "voice" ? "opacity-50 pointer-events-none" : ""
          }`}
        >
          <h2 className="text-2xl font-bold text-red-700 mb-4">Input Testo</h2>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setInputMode("text");
            }}
            placeholder="Scrivi il testo per l'analisi..."
            className="w-full h-40 p-4 rounded-lg bg-white/80 focus:outline-none shadow-inner"
          ></textarea>
          <motion.button
            onClick={handleTextSubmit}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="mt-4 w-full py-3 bg-yellow-400 text-red-900 font-bold text-lg rounded-full shadow-lg hover:bg-yellow-500 transition"
          >
            Invia <FaArrowRight className="inline ml-2" />
          </motion.button>
        </motion.div>

        {/* Card Input Voce */}
        <motion.div
          initial={{ opacity: 0, x: 80 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className={`backdrop-blur-md bg-white/30 p-8 rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition ${
            inputMode === "text" ? "opacity-50 pointer-events-none" : ""
          }`}
        >
          <h2 className="text-2xl font-bold text-yellow-600 mb-4">Input Voce</h2>
          <motion.button
            onClick={handleVoiceToggle}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className={`w-full py-4 rounded-full text-white font-bold text-xl shadow-lg ${
              isRecording ? "bg-red-700 animate-pulse" : "bg-red-600 hover:bg-red-700"
            }`}
          >
            {isRecording ? (
              <>
                <FaMicrophoneSlash className="inline mr-2" /> Ferma Registrazione
              </>
            ) : (
              <>
                <FaMicrophone className="inline mr-2" /> Avvia Registrazione
              </>
            )}
          </motion.button>

          {isRecording && (
            <p className="mt-4 text-gray-700 italic text-center">
              🎤 Sto ascoltando... parla ora
            </p>
          )}
          {partialText && !isRecording && (
            <p className="mt-4 text-gray-600 italic text-center">
              Trascrizione: {partialText}
            </p>
          )}
        </motion.div>
      </section>

      {/* RISPOSTA */}
      {(response || loading) && (
        <motion.section
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto text-center bg-white/40 backdrop-blur-md p-8 rounded-2xl shadow-xl mt-8 border border-white/20 whitespace-pre-line"
        >
          <h3 className="text-2xl font-bold text-red-700 mb-4">Risultato Analisi</h3>
          {loading ? (
            <div className="flex justify-center">
              <div className="w-8 h-8 border-4 border-red-700 border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : (
            <p className="text-lg text-gray-800">{response}</p>
          )}
        </motion.section>
      )}

      {/* FOOTER */}
      <footer id="contatti" className="bg-gray-900 text-gray-200 py-12 mt-16">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-lg mb-6">© 2025 Eustachio Fontana - Tutti i diritti riservati</p>
        </div>
      </footer>
    </div>
  );
}

export default Analyzer;