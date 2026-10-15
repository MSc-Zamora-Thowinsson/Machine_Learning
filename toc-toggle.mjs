// Botón para ocultar / mostrar el índice lateral (TOC) en mystmd.
// Se carga desde footer.md con la directiva {anywidget}. El estilo está en estilos.css.
const KEY = 'myst-toc-oculto';

function aplicarEstadoGuardado() {
  try {
    if (localStorage.getItem(KEY) === '1') {
      document.documentElement.classList.add('toc-oculto');
    }
  } catch (e) {}
}

function crearBoton() {
  if (document.getElementById('toc-toggle-btn')) return;

  const btn = document.createElement('button');
  btn.id = 'toc-toggle-btn';
  btn.type = 'button';
  btn.title = 'Ocultar / mostrar índice';
  btn.setAttribute('aria-label', 'Ocultar o mostrar el índice');
  btn.innerHTML =
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg>';

  btn.addEventListener('click', () => {
    const oculto = document.documentElement.classList.toggle('toc-oculto');
    try {
      localStorage.setItem(KEY, oculto ? '1' : '0');
    } catch (e) {}
  });

  const barra = document.querySelector('.myst-top-nav-bar');
  if (barra) {
    barra.insertBefore(btn, barra.firstChild);
  } else {
    btn.classList.add('flotante');
    document.body.appendChild(btn);
  }
}

export default {
  render() {
    aplicarEstadoGuardado();
    crearBoton();
  },
};
