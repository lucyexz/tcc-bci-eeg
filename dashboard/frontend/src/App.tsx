import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Overview from "./pages/Overview";
import ModelDetail from "./pages/ModelDetail";
import Comparison from "./pages/Comparison";

export default function App() {
  return (
    <BrowserRouter>
      <nav className="navbar">
        <span className="navbar-brand">BCI EEG</span>
        <div className="navbar-links">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Overview
          </NavLink>
          <NavLink to="/comparison" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Comparação
          </NavLink>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/model/:name" element={<ModelDetail />} />
          <Route path="/comparison" element={<Comparison />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
