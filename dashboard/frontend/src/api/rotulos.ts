// Nomes de exibição das figuras e tabelas (os arquivos não têm acento).

const ROTULOS: Record<string, string> = {
  // figuras dos modelos e do comparativo
  resultado: "Resultado",
  resultado_principal: "Resultado principal",
  matriz_confusao: "Matriz de confusão (hold-out)",
  curvas_treino: "Curvas de treino",
  permutacao: "Permutação de rótulos",
  calibracao: "Calibração",
  xai_eletrodo_banda: "Integrated Gradients por eletrodo × banda",
  xai_tempo: "Integrated Gradients ao longo da janela",
  atencao: "Atenção do [CLS] ao longo da janela",
  por_participante: "Por participante",
  divisao_blocos: "Divisão dos registros em treino, validação e teste",
  // EDA
  duracao_blocos: "Duração dos blocos",
  distribuicao_log: "Distribuição dos valores (log)",
  correlacao: "Correlação entre as features",
  valores_extremos: "Valores extremos",
  autocorrelacao: "Autocorrelação",
  poder_estatistico: "Poder estatístico",
  variancia_bloco_classe: "Variância entre blocos × entre classes",
  deriva_sessao: "Deriva ao longo da sessão",
  anova_classes: "ANOVA entre classes",
  espectro_classes: "Espectro médio por classe",
  tsne_global: "t-SNE (padronização global)",
  tsne_por_participante: "t-SNE (padronização por participante)",
  registros_por_classe: "Registros por classe",
  qualidade: "Qualidade dos dados",
  valores_nao_positivos: "Valores não positivos",
  estatisticas_descritivas: "Estatísticas descritivas",
  balanco_classes: "Balanço das classes",
  mediana_log: "Mediana do log da potência",
  tamanho_amostral_efetivo: "Tamanho amostral efetivo",
  efeito_classe: "Efeito da classe",
};

// "eda_05_autocorrelacao" / "03_curvas_treino" / "mediana_log" → rótulo em português
export function rotulo(nome: string): string {
  const chave = nome.replace(/^(eda_)?\d+_/, "");
  if (ROTULOS[chave]) return ROTULOS[chave];
  const texto = chave.replace(/_/g, " ");
  return texto.charAt(0).toUpperCase() + texto.slice(1);
}
