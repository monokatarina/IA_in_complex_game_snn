import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.game.config import Config

# Importações adicionais para suporte ao SNN
from models.utils.helpers import calculate_distance, normalize_position
import snntorch as snn
import snntorch.functional as SF

class EnhancedSNN(nn.Module):
    def __init__(self, with_curiosity=None):
        super().__init__()
        
        # 1. Normalização de entrada
        self.input_norm = nn.LayerNorm(Config.STATE_SIZE)
        
        # 2. Processamento visual (STATE_SIZE → 512)
        self.visual_processing = nn.Sequential(
            nn.Linear(Config.STATE_SIZE, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.1))
        
        # 3. Mecanismo de atenção
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # Deve corresponder à saída do visual_processing
            num_heads=8,
            dropout=0.1,
            batch_first=True)
        
        # 4. Blocos residuais com dimensões corrigidas
        self.res_block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            nn.Dropout(0.2))
            
        self.res_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            nn.Dropout(0.2))
            
        self.res_block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Dropout(0.1))
        
        # 5. Camadas SNN (corrigidas para usar fc_snn em vez de fc)
        self.fc_snn0 = nn.Linear(64, 64)
        self.lif0 = snn.Leaky(beta=0.95, threshold=0.8, reset_mechanism="zero")
        
        self.fc_snn1 = nn.Linear(64, 64)
        self.lif1 = snn.Leaky(beta=0.92, threshold=0.85, reset_mechanism="zero")
        
        self.fc_snn2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.9, threshold=0.9, reset_mechanism="zero")
        
        self.fc_snn3 = nn.Linear(32, 32)
        self.lif3 = snn.Leaky(beta=0.85, threshold=0.95, reset_mechanism="zero")
        
        # 6. Camada de saída
        self.fc_out = nn.Linear(32, 5)
        
    def forward(self, x: torch.Tensor, time_window: int = 10) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(1) != Config.STATE_SIZE:
            raise ValueError(f"Tamanho do estado inválido: esperado {Config.STATE_SIZE}, recebido {x.size(1)}")
            
        # Normalização de entrada
        x = self.input_norm(x)
        
        # Processamento visual
        visual_out = self.visual_processing(x)
        
        # Atenção
        attn_out, _ = self.attention(
            visual_out.unsqueeze(1,), # Adiciona dimensão de batch 
            visual_out.unsqueeze(1),
            visual_out.unsqueeze(1))
        attn_out = attn_out.squeeze(1) 
        
        # Blocos residuais (corrigido o skip connection)
        res1 = self.res_block1(attn_out)
        res1 = res1 + attn_out[:, :256]  # Skip connection ajustada
        
        res2 = self.res_block2(res1)
        res2 = res2 + res1[:, :128]  # Skip connection ajustada
        
        res3 = self.res_block3(res2)
        res3 = res3 + res2[:, :64]  # Skip connection ajustada
        
        # Processamento SNN
        mem0 = self.lif0.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spikes = torch.zeros(x.size(0), 5, device=x.device)
        
        for _ in range(time_window):
            x0 = self.fc_snn0(res3)
            spk0, mem0 = self.lif0(x0, mem0)
            
            x1 = self.fc_snn1(spk0)
            spk1, mem1 = self.lif1(x1, mem1)
            
            x2 = self.fc_snn2(spk1)
            spk2, mem2 = self.lif2(x2, mem2)
            
            x3 = self.fc_snn3(spk2)
            spk3, mem3 = self.lif3(x3, mem3)
            
            spikes += torch.sigmoid(self.fc_out(spk3))
            
        return spikes / time_window


class CuriosityModule(nn.Module):
    """Módulo de curiosidade para aprendizado intrínseco"""
    
    def __init__(self, input_size: int, hidden_size: int = 32):
        """Inicializa o módulo de curiosidade
        
        Args:
            input_size: Tamanho do estado de entrada
            hidden_size: Tamanho da camada oculta
        """
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(input_size + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, 
            next_state: torch.Tensor) -> torch.Tensor:
        """Calcula recompensa de curiosidade
        
        Args:
            state: Estado atual
            action: Ação tomada
            next_state: Próximo estado
            
        Returns:
            Recompensa de curiosidade
        """
        action_onehot = torch.zeros(action.size(0), 5, device=action.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        concatenated = torch.cat([state, action_onehot], dim=1)
        predicted_next_state = self.forward_model(concatenated)
        return F.mse_loss(predicted_next_state, next_state, reduction='none').mean(1)