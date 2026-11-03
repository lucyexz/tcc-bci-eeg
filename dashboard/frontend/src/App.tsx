import { BrowserRouter, Navigate, NavLink, Route, Routes, useParams } from "react-router-dom";
import { MODELOS, NOMES } from "./api/client";
import Dados from "./pages/Dados";
import Modelo from "./pages/Modelo";
import VisaoGeral from "./pages/VisaoGeral";
import { useTema } from "./tema";

const classe = ({ isActive }: { isActive: boolean }) => (isActive ? "nav-link active" : "nav-link");

function BotaoTema() {
  const { tema, alternar } = useTema();
  return (
    <button className="botao-tema" onClick={alternar} title="Alternar tema claro/escuro">
      {tema === "claro" ? "☾ Escuro" : "☀ Claro"}
    </button>
  );
}

// Endereço antigo da página do simulador: agora ele fica no topo da página do modelo
function RedirecionaSimulador() {
  const { nome = "" } = useParams();
  return <Navigate to={`/modelo/${nome}#simulador`} replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      {/* Navegação na lateral; em tela estreita (até 900 px) ela vai para o topo (CSS) */}
      <div className="app">
        <nav className="navbar">
          <span className="navbar-brand">BCI EEG</span>
          <div className="navbar-links">
            <span className="nav-grupo">Resultados</span>
            <NavLink to="/" end className={classe}>
              Visão geral
            </NavLink>
            <span className="nav-grupo">Modelos</span>
            {MODELOS.map((m) => (
              <NavLink key={m} to={`/modelo/${m}`} className={classe}>
                {NOMES[m]}
              </NavLink>
            ))}
            <span className="nav-grupo">Dataset</span>
            <NavLink to="/dados" className={classe}>
              Dados
            </NavLink>
          </div>
          <div className="navbar-acoes">
            <BotaoTema />
          </div>
        </nav>
        <div className="area-conteudo">
          <main className="main-content">
            <Routes>
              <Route path="/" element={<VisaoGeral />} />
              <Route path="/modelo/:nome" element={<Modelo />} />
              <Route path="/modelo/:nome/simulador" element={<RedirecionaSimulador />} />
              <Route path="/dados" element={<Dados />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
