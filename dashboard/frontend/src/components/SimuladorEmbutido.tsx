// Simulador no topo da página de cada modelo: a mesma simulação do `python main.py simulador` (Pygame),
// com as mesmas opções do terminal. Só carrega os dados ao clicar em Iniciar (nos Transformers isso
// carrega o TensorFlow) e para de desenhar quando sai da tela.
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  buscarSimulador,
  EtapaAusente,
  NOMES,
  PARADIGMAS,
  PARTICIPANTES,
  type DadosSimulador,
} from "../api/client";
import { CONFIG } from "../simulador/config";
import { ControladorMock, ControladorModelo } from "../simulador/controladores";
import { Simulacao, type EstadoControles } from "../simulador/loop";

const FPS = [30, 60, 120];
const COM_ATENCAO = ["transformer", "cnn_transformer"];

interface Carga {
  chave: string;
  dados: DadosSimulador | null;
  ausente: string | null;
  erro: string | null;
}

export default function SimuladorEmbutido({ modelo }: { modelo: string }) {
  const [iniciado, setIniciado] = useState(false);
  const [paradigma, setParadigma] = useState<string>("especialista");
  const [participante, setParticipante] = useState<string>("user_a");
  const [velocidadeTexto, setVelocidadeTexto] = useState("8");
  const [fps, setFps] = useState(60);
  const [mock, setMock] = useState(false);
  const [carga, setCarga] = useState<Carga | null>(null);
  const [controles, setControles] = useState<EstadoControles>({ pausado: false, congelado: false });
  const caixaRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Simulacao | null>(null);
  const visivelRef = useRef(true);

  const v = Number(velocidadeTexto.replace(",", "."));
  const velocidade = Number.isFinite(v) && v > 0 ? Math.min(v, 1000) : 8;
  const chave = `${modelo}|${paradigma}|${participante}`;
  const dados = !mock && carga?.chave === chave ? carga.dados : null;

  // Dados do teste: só depois de Iniciar e só no modo real (em cache depois da primeira vez)
  useEffect(() => {
    if (!iniciado || mock) return;
    let ativo = true;
    buscarSimulador(modelo, paradigma, participante)
      .then((d) => ativo && setCarga({ chave, dados: d, ausente: null, erro: null }))
      .catch((e: Error) =>
        ativo &&
        setCarga({
          chave,
          dados: null,
          ausente: e instanceof EtapaAusente ? e.message : null,
          erro: e instanceof EtapaAusente ? null : e.message,
        }),
      );
    return () => {
      ativo = false;
    };
  }, [iniciado, modelo, paradigma, participante, mock, chave]);

  // Qualquer mudança de configuração recomeça a simulação do início
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!iniciado || !canvas || (!mock && !dados)) return;
    const sim = new Simulacao(
      canvas,
      {
        controlador: mock ? new ControladorMock(CONFIG.mock_class_hold_ms) : new ControladorModelo(dados!),
        useMock: mock,
        modelName: `${NOMES[modelo] ?? modelo} · ${paradigma}`,
        fps,
        // Como no Pygame: no modo real, 1000 / velocidade; no mock o intervalo fica em 1000 ms
        intervaloMs: mock ? 1000 : Math.max(1, Math.round(1000 / velocidade)),
      },
      setControles,
    );
    simRef.current = sim;
    setControles({ pausado: false, congelado: false });
    sim.iniciar();
    if (!visivelRef.current) sim.suspender();
    return () => {
      sim.parar();
      simRef.current = null;
    };
  }, [iniciado, dados, mock, fps, velocidade, modelo, paradigma]);

  // Fora da tela, o loop para de desenhar; volta quando o simulador reaparece
  useEffect(() => {
    const caixa = caixaRef.current;
    if (!caixa || typeof IntersectionObserver === "undefined") return;
    const obs = new IntersectionObserver(([e]) => {
      visivelRef.current = e.isIntersecting;
      if (e.isIntersecting) simRef.current?.retomar();
      else simRef.current?.suspender();
    });
    obs.observe(caixa);
    return () => obs.disconnect();
  }, []);

  const teclas = (e: KeyboardEvent) => {
    const sim = simRef.current;
    if (!sim) return;
    if (e.code === "Space") sim.alternarPausa();
    else if (e.code === "KeyF") sim.alternarCongelar();
    else if (e.code === "KeyR") sim.resetar();
    else return;
    e.preventDefault();
  };

  const falhou = carga?.chave === chave && !mock ? carga : null;
  const rodando = iniciado && (mock || !!dados);
  const carregando = iniciado && !mock && !dados && !(falhou?.ausente || falhou?.erro);

  return (
    <div className="sim-embutido" ref={caixaRef}>
      <div className="sim-config">
        <label>
          Paradigma
          <select className="seletor" value={paradigma} disabled={mock} onChange={(e) => setParadigma(e.target.value)}>
            {PARADIGMAS.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label>
          Participante
          <select className="seletor" value={participante} disabled={mock}
            onChange={(e) => setParticipante(e.target.value)}>
            {PARTICIPANTES.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </label>
        <label title="Janelas por segundo. 8 = o ritmo em que o dataset foi gravado (tempo real).">
          Janelas/s
          <input className="seletor sim-numero" type="number" min={0.5} step={0.5} value={velocidadeTexto}
            disabled={mock} onChange={(e) => setVelocidadeTexto(e.target.value)} />
        </label>
        <label>
          FPS
          <select className="seletor" value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            {FPS.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </label>
        <label className="sim-check">
          <input type="checkbox" checked={mock} onChange={(e) => setMock(e.target.checked)} />
          Sem modelo
        </label>
      </div>

      {!iniciado ? (
        <div className="sim-inicio">
          <p>
            O carro segue a classe prevista em cada janela do teste do hold-out (o fim da sessão, que o modelo não viu
            no treino), na ordem em que o EEG foi gravado. Modelo da semente 42, como no{" "}
            <code>python main.py simulador</code>.
          </p>
          <button className="sim-iniciar" onClick={() => setIniciado(true)}>
            ▶ Iniciar
          </button>
        </div>
      ) : (
        <>
          <div className="sim-botoes">
            <button className="botao-tema" disabled={!rodando} onClick={() => simRef.current?.alternarPausa()}>
              {controles.pausado ? "Retomar" : "Pausar"} <kbd>Espaço</kbd>
            </button>
            <button className="botao-tema" disabled={!rodando} onClick={() => simRef.current?.alternarCongelar()}>
              {controles.congelado ? "Descongelar" : "Congelar dados"} <kbd>F</kbd>
            </button>
            <button className="botao-tema" disabled={!rodando} onClick={() => simRef.current?.resetar()}>
              Resetar <kbd>R</kbd>
            </button>
            {dados && (
              <span className="muted">
                {dados.n} janelas (T = {dados.T}); acerto no trecho todo:{" "}
                {(100 * dados.acerto_total).toFixed(1).replace(".", ",")}%
              </span>
            )}
          </div>

          {falhou?.ausente && (
            <div className="aviso">
              <strong>Sem resultados ainda.</strong>
              <p>{falhou.ausente}</p>
            </div>
          )}
          {falhou?.erro && <div className="error-msg">Erro: {falhou.erro}</div>}
          {carregando && (
            <div className="sim-inicio">
              <p className="muted">
                Carregando…{" "}
                {COM_ATENCAO.includes(modelo) &&
                  "Na primeira vez, o modelo é carregado para calcular a atenção (alguns segundos)."}
              </p>
            </div>
          )}

          <canvas
            ref={canvasRef}
            className="sim-canvas"
            width={CONFIG.window_width * 2}
            height={CONFIG.window_height * 2}
            tabIndex={0}
            onKeyDown={teclas}
            style={{ display: rodando ? "block" : "none" }}
            aria-label="Simulação: o carro segue a classe prevista pelo modelo"
          />
          {rodando && <p className="muted sim-nota">Clique na simulação para usar as teclas.</p>}
        </>
      )}
    </div>
  );
}
