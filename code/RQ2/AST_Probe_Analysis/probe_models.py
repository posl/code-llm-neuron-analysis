import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BaseProbe(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super(BaseProbe, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dtype = dtype

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.classifier = nn.Linear(input_dim, output_dim)
        
        self.classifier = self.classifier.to(dtype=dtype)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(representations)

class ASTNodeProbe(BaseProbe):
    def __init__(self,
                 input_dim: int,
                 num_ast_types: int,
                 hidden_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super(ASTNodeProbe, self).__init__(
            input_dim=input_dim,
            output_dim=num_ast_types,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype
        )
        self.num_ast_types = num_ast_types

class LanguageProbe(BaseProbe):
    
    def __init__(self,
                 input_dim: int,
                 num_languages: int,
                 hidden_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super(LanguageProbe, self).__init__(
            input_dim=input_dim,
            output_dim=num_languages,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype
        )
        self.num_languages = num_languages

class MultiTaskProbe(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 num_ast_types: int,
                 num_languages: int,
                 shared_hidden_dim: Optional[int] = None,
                 task_hidden_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 use_shared_encoder: bool = True,
                 dtype: torch.dtype = torch.float32):
        super(MultiTaskProbe, self).__init__()

        self.input_dim = input_dim
        self.num_ast_types = num_ast_types
        self.num_languages = num_languages
        self.use_shared_encoder = use_shared_encoder
        self.dtype = dtype

        if use_shared_encoder and shared_hidden_dim is not None:
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, shared_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            encoder_output_dim = shared_hidden_dim
        else:
            self.shared_encoder = nn.Identity()
            encoder_output_dim = input_dim
        self.ast_classifier = self._build_classifier(
            encoder_output_dim, num_ast_types, task_hidden_dim, dropout_rate
        )
        self.language_classifier = self._build_classifier(
            encoder_output_dim, num_languages, task_hidden_dim, dropout_rate
        )

        self.shared_encoder = self.shared_encoder.to(dtype=dtype)
        self.ast_classifier = self.ast_classifier.to(dtype=dtype)
        self.language_classifier = self.language_classifier.to(dtype=dtype)

        self._init_weights()
    
    def _build_classifier(self, input_dim: int, output_dim: int, 
                         hidden_dim: Optional[int], dropout_rate: float) -> nn.Module:
        if hidden_dim is not None:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            return nn.Linear(input_dim, output_dim)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_encoder(representations)
        ast_logits = self.ast_classifier(shared_features)
        language_logits = self.language_classifier(shared_features)
        return ast_logits, language_logits

class GradientReversalFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None

def gradient_reversal_layer(lambda_adv):
    def gradient_reversal(x):
        return GradientReversalFunction.apply(x, lambda_adv)
    return gradient_reversal

class AdversarialProbe(nn.Module):
    
    def __init__(self, 
                 input_dim: int,
                 num_ast_types: int,
                 num_languages: int,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.1,
                 lambda_adv: float = 0.1):
        super(AdversarialProbe, self).__init__()
        
        self.lambda_adv = lambda_adv
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.ast_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_ast_types)
        )
        
        self.language_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_languages)
        )
        
        self.gradient_reversal = gradient_reversal_layer(lambda_adv)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, representations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(representations)
        ast_logits = self.ast_classifier(features)
        reversed_features = self.gradient_reversal(features)
        language_logits = self.language_discriminator(reversed_features)
        return ast_logits, language_logits

class ProbeEnsemble(nn.Module):
    
    def __init__(self, probes: list, ensemble_method: str = "average"):
        super(ProbeEnsemble, self).__init__()
        
        self.probes = nn.ModuleList(probes)
        self.ensemble_method = ensemble_method
        self.num_probes = len(probes)
        
        if ensemble_method == "weighted":
            self.weights = nn.Parameter(torch.ones(self.num_probes) / self.num_probes)
    
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        predictions = []
        for probe in self.probes:
            pred = probe(representations)
            predictions.append(pred)
        
        if self.ensemble_method == "average":
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            weighted_preds = [w * pred for w, pred in zip(weights, predictions)]
            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
        elif self.ensemble_method == "voting":
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        return ensemble_pred