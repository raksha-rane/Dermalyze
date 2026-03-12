export interface ClassInfo {
  id: string;
  name: string;
  description: string;
  riskLevel: string;
  commonIn: string;
  keyFeatures: string;
}

export type RiskSeverity = 'low' | 'moderate' | 'high' | 'critical';

export const classInfoMap: Record<string, ClassInfo> = {
  akiec: {
    id: 'akiec',
    name: 'Actinic Keratosis',
    description: 'A precancerous lesion caused by chronic sun exposure; may progress to squamous cell carcinoma if untreated.',
    riskLevel: 'Moderate to High',
    commonIn: 'Older adults, fair skin, heavily sun-exposed areas',
    keyFeatures: 'Rough or scaly erythematous patches, persistent tenderness, superficial crusting',
  },
  bcc: {
    id: 'bcc',
    name: 'Basal Cell Carcinoma',
    description: 'The most common skin cancer, usually slow-growing with low metastatic potential but can be locally destructive.',
    riskLevel: 'High',
    commonIn: 'Adults with cumulative UV exposure, face and neck',
    keyFeatures: 'Pearly papule, rolled border, telangiectasia, possible central ulceration',
  },
  bkl: {
    id: 'bkl',
    name: 'Benign Keratosis',
    description: 'A benign keratinocytic lesion group, often including seborrheic keratosis and lichenoid keratosis.',
    riskLevel: 'Low',
    commonIn: 'Middle-aged and older adults',
    keyFeatures: 'Waxy or verrucous surface, well-circumscribed borders, variable pigmentation',
  },
  df: {
    id: 'df',
    name: 'Dermatofibroma',
    description: 'A benign fibrohistiocytic skin nodule that is generally stable and non-malignant.',
    riskLevel: 'Low',
    commonIn: 'Young to middle-aged adults, frequently lower extremities',
    keyFeatures: 'Firm papule or nodule, dimple sign, peripheral pigment network',
  },
  mel: {
    id: 'mel',
    name: 'Melanoma',
    description: 'An aggressive malignant tumor of melanocytes with significant metastatic potential if not detected early.',
    riskLevel: 'Critical',
    commonIn: 'Any adult population; risk increases with UV damage and atypical nevi history',
    keyFeatures: 'Asymmetry, border irregularity, color variegation, diameter growth, evolution over time',
  },
  nv: {
    id: 'nv',
    name: 'Melanocytic Nevus',
    description: 'A common benign melanocytic lesion that is typically stable over time.',
    riskLevel: 'Low',
    commonIn: 'All age groups, often appearing in childhood and early adulthood',
    keyFeatures: 'Symmetric shape, regular borders, uniform pigmentation, stable appearance',
  },
  vasc: {
    id: 'vasc',
    name: 'Vascular Lesion',
    description: 'A benign vascular proliferation such as angioma; usually non-cancerous.',
    riskLevel: 'Low',
    commonIn: 'Adults, trunk and extremities',
    keyFeatures: 'Red to violaceous coloration, lacunar pattern, blanching in some cases',
  },
};

export const getRiskSeverity = (riskLevel: string): RiskSeverity => {
  const value = riskLevel.toLowerCase();

  if (value.includes('critical')) return 'critical';
  if (value.includes('high')) return 'high';
  if (value.includes('moderate')) return 'moderate';
  return 'low';
};

export const getRiskBadgeStyles = (severity: RiskSeverity) => {
  const styles: Record<
    RiskSeverity,
    { bg: string; text: string; border: string; dot: string }
  > = {
    critical: {
      bg: 'bg-red-50',
      text: 'text-red-700',
      border: 'border-red-200',
      dot: 'bg-red-500',
    },
    high: {
      bg: 'bg-orange-50',
      text: 'text-orange-700',
      border: 'border-orange-200',
      dot: 'bg-orange-500',
    },
    moderate: {
      bg: 'bg-amber-50',
      text: 'text-amber-700',
      border: 'border-amber-200',
      dot: 'bg-amber-500',
    },
    low: {
      bg: 'bg-emerald-50',
      text: 'text-emerald-700',
      border: 'border-emerald-200',
      dot: 'bg-emerald-500',
    },
  };

  return styles[severity];
};

export const getRiskLabel = (severity: RiskSeverity): string => {
  if (severity === 'critical') return 'Critical Risk';
  if (severity === 'high') return 'High Risk';
  if (severity === 'moderate') return 'Moderate Risk';
  return 'Low Risk';
};

export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 85) return 'text-emerald-600';
  if (confidence >= 70) return 'text-teal-600';
  if (confidence >= 50) return 'text-amber-600';
  return 'text-red-600';
};
