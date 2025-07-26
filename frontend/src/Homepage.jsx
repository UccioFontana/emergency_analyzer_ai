import { motion, useScroll, useTransform } from "framer-motion";
import { useRef, useEffect } from "react";
import Particles from "react-tsparticles";
import bgImage from "./assets/vigilimatera.jpg"; // tua immagine
import { Link } from "react-router-dom"; // AGGIUNGI QUESTO in Homepage.jsx


function Homepage() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  // Animazioni parallax
  const bgY = useTransform(scrollYProgress, [0, 1], ["0%", "30%"]);
  const bgScale = useTransform(scrollYProgress, [0, 1], [1.2, 1]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  // Effetto fumo canvas
  useEffect(() => {
    const canvas = document.getElementById("smokeCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particles = Array(50).fill().map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 20 + 20,
      dx: (Math.random() - 0.5) * 0.5,
      dy: (Math.random() - 0.5) * 0.5,
      alpha: Math.random() * 0.5 + 0.2
    }));

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        ctx.beginPath();
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
        gradient.addColorStop(0, `rgba(255,255,255,${p.alpha})`);
        gradient.addColorStop(1, "transparent");
        ctx.fillStyle = gradient;
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        p.x += p.dx;
        p.y += p.dy;
        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
      });
      requestAnimationFrame(draw);
    }
    draw();
  }, []);

  return (
    <div className="bg-gray-50 text-gray-900 font-sans overflow-x-hidden">
      {/* HEADER */}
      <header className="fixed top-0 left-0 w-full z-50 bg-gradient-to-r from-red-700 to-red-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-extrabold tracking-wide uppercase">
            Vigili del Fuoco <span className="text-yellow-400">Matera</span>
          </h1>
          <nav className="flex gap-8 text-lg font-semibold">
            <a href="#servizi" className="hover:text-yellow-400 transition">Servizi</a>
            <a href="#interventi" className="hover:text-yellow-400 transition">Schede</a>
            <a href="#contatti" className="hover:text-yellow-400 transition">Contatti</a>
          </nav>
        </div>
      </header>

      {/* HERO */}
      <section
        ref={ref}
        className="relative h-screen flex flex-col justify-center items-center text-center overflow-hidden"
      >
        {/* BACKGROUND IMMERSIVO */}
        <motion.img
          src={bgImage}
          alt="Vigili del Fuoco Matera"
          className="absolute inset-0 w-full h-full object-cover"
          style={{ y: bgY, scale: bgScale }}
        />

        {/* OVERLAY ROSSO */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: "rgba(139,0,0,0.6)",
            backdropFilter: "blur(2px)",
          }}
          initial={{ opacity: 0.6 }}
          animate={{ opacity: [0.4, 0.6, 0.4] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        ></motion.div>

        {/* PARTICELLE */}
        <Particles
          className="absolute inset-0 z-0"
          options={{
            background: { color: { value: "transparent" } },
            fpsLimit: 60,
            particles: {
              number: { value: 40 },
              color: { value: "#FFD700" },
              size: { value: 2 },
              move: { enable: true, speed: 1.5 },
              opacity: { value: 0.8 },
            },
          }}
        />

      {/* CANVAS FUMO */}
      <canvas id="smokeCanvas" className="absolute inset-0 z-0"></canvas>

        {/* CONTENUTO HERO */}
        <motion.div
          className="relative z-10 max-w-4xl px-6"
          style={{ opacity: heroOpacity }}
        >
          <motion.h1
            className="text-white text-5xl md:text-8xl font-extrabold drop-shadow-[0_8px_10px_rgba(0,0,0,0.8)] leading-tight"
            whileHover={{ rotateX: 8, rotateY: 8, scale: 1.05 }}
            transition={{ type: "spring", stiffness: 200 }}
          >
            Proteggiamo la tua vita
          </motion.h1>

          <motion.p
            className="mt-6 text-lg md:text-2xl text-gray-200 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.5 }}
          >
            Vigili del Fuoco Matera: sicurezza, prevenzione e intervento rapido.
          </motion.p>

          <motion.a
            href="#servizi"
            className="inline-block mt-10 px-10 py-4 bg-yellow-400 text-red-900 font-bold rounded-full text-xl shadow-lg hover:bg-yellow-500 transition-transform"
            whileHover={{ scale: 1.15 }}
            whileTap={{ scale: 0.95 }}
          >
            Scopri i servizi
          </motion.a>
        </motion.div>



        {/* SCROLL INDICATOR */}
        <motion.div
          className="absolute bottom-6 text-white text-sm flex flex-col items-center"
          animate={{ y: [0, 10, 0] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
        >
          Scrolla
          <div className="w-1 h-8 bg-white rounded-full mt-2"></div>
        </motion.div>
      </section>

      {/* SEZIONE SERVIZI */}
      <motion.section
        id="servizi"
        className="relative py-24 bg-gray-100"
        initial={{ opacity: 0, y: 80 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <div className="max-w-7xl mx-auto px-6 text-center mb-16">
          <h2 className="text-4xl font-bold text-red-700 mb-4">I Nostri Servizi</h2>
          <p className="text-lg text-gray-600">
            Soluzioni avanzate per garantire sicurezza e prevenzione.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 max-w-6xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, x: -100 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            whileHover={{ scale: 1.08, rotate: 1 }}
            className="backdrop-blur-md bg-white/30 p-10 rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition"
          >
            <h3 className="text-3xl font-bold text-red-700 mb-4">Previsione Emergenze</h3>
            <p className="text-gray-700 mb-6">
              Analizziamo rischi e prevediamo emergenze con strumenti avanzati per
              ridurre al minimo i pericoli.
            </p>
            <Link to="/analyzer" className="text-red-700 font-semibold hover:underline">
                Scopri di più →
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 100 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            whileHover={{ scale: 1.08, rotate: 1 }}
            className="backdrop-blur-md bg-white/30 p-10 rounded-2xl shadow-xl border border-white/20 hover:shadow-2xl transition"
          >
            <h3 className="text-3xl font-bold text-yellow-600 mb-4">Schede Intervento</h3>
            <p className="text-gray-700 mb-6">
              Tutte le informazioni operative per interventi rapidi e sicuri in caso di emergenza.
            </p>
            <a href="#" className="text-yellow-600 font-semibold hover:underline">
              Visualizza le schede →
            </a>
          </motion.div>
        </div>
      </motion.section>

      {/* FOOTER */}
      <footer id="contatti" className="bg-gray-900 text-gray-200 py-12 mt-16">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-lg mb-6">© 2025 Vigili del Fuoco Matera - Tutti i diritti riservati</p>
          <div className="flex justify-center gap-8 text-xl">
            <a href="#" className="hover:text-red-500 transition">Facebook</a>
            <a href="#" className="hover:text-red-500 transition">Twitter</a>
            <a href="#" className="hover:text-red-500 transition">Instagram</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Homepage;