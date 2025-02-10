import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from node2vec import Node2Vec
import networkx as nx
from sklearn.model_selection import train_test_split

def create_country_graph(countries):
    G = nx.DiGraph(directed=True)
    edges = []
    for i in range(len(countries)):
        last_letter = countries[i][-1].upper()
        for j in range(len(countries)):
            if i != j:
                first_letter = countries[j][0].upper()
                if last_letter == first_letter:
                    edges.append([countries[i], countries[j]])
    
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G, edges

class UnsupervisedNode2Vec:
    def __init__(self, graph, embedding_dim=128, walk_length=30, num_walks=200, p=1, q=1):
        self.original_graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.embeddings = None
        
    def create_training_graph(self, train_edges):
        self.train_graph = nx.DiGraph()
        self.train_graph.add_nodes_from(self.original_graph.nodes())
        self.train_graph.add_edges_from(train_edges)
        
    def train_embeddings(self):
        graph_str = nx.relabel_nodes(self.train_graph, lambda x: str(x))
        
        node2vec = Node2Vec(
            graph_str,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=4
        )
        
        model = node2vec.fit(window=10, min_count=1)
        self.embeddings = {node: model.wv[str(node)] for node in self.original_graph.nodes()}
    
    def get_edge_score(self, edge):
        src_emb = self.embeddings[edge[0]]
        dst_emb = self.embeddings[edge[1]]
        similarity = np.dot(src_emb, dst_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(dst_emb))
        return similarity

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

class UnsupervisedGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super(UnsupervisedGAE, self).__init__(encoder, decoder)
        
    def get_edge_score(self, z, edge):
        src_emb = z[edge[0]]
        dst_emb = z[edge[1]]
        return torch.sigmoid(torch.sum(src_emb * dst_emb))

def prepare_masked_pyg_data(graph, node_features, countries):
    node_to_idx = {node: idx for idx, node in enumerate(countries)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    

    edge_index = torch.tensor([[node_to_idx[e[0]], node_to_idx[e[1]]] 
                             for e in graph.edges()], dtype=torch.long).t()
    x = torch.tensor(node_features, dtype=torch.float)
    

    data = Data(x=x, edge_index=edge_index)
    

    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
    
    return data, node_to_idx, idx_to_node

def train_gae(model, data, optimizer, num_epochs=100):
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)
        
        if epoch % 10 == 0:
            auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {auc:.4f}, AP: {ap:.4f}')
            
        loss.backward()
        optimizer.step()
    
    return model

def main():
    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", 
        "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", 
        "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", 
        "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", 
        "Chile", "China", "Colombia", "Comoros", "Congo", "Democratic Republic of the Congo", "Costa Rica", "Croatia", 
        "Cuba", "Cyprus", "Czechia", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", 
        "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", 
        "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", 
        "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", 
        "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", 
        "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", 
        "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", 
        "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", 
        "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", 
        "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", 
        "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", 
        "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", 
        "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", 
        "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", 
        "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", 
        "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", 
        "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",

        "Tokyo", "Delhi", "Shanghai", "Dhaka", "Sao Paulo", "Cairo", "Mexico City", "Beijing", "Mumbai", "Osaka",
        "Chongqing", "Karachi", "Kinshasa", "Lagos", "Istanbul", "Buenos Aires", "Kolkata", "Manila", "Guangzhou", "Tianjin",
        "Lahore", "Bangalore", "Rio de Janeiro", "Shenzhen", "Moscow", "Chennai", "Bogota", "Jakarta", "Lima", "Paris",
        "Bangkok", "Hyderabad", "Seoul", "Nanjing", "Chengdu", "London", "Luanda", "Tehran", "Ho Chi Minh City", "Nagoya",
        "Xi-an", "Ahmedabad", "Wuhan", "Kuala Lumpur", "Hangzhou", "Suzhou", "Surat", "Dar es Salaam", "New York City", "Baghdad",
        "Shenyang", "Riyadh", "Hong Kong", "Foshan", "Dongguan", "Pune", "Santiago", "Haerbin", "Madrid", "Khartoum",
        "Toronto", "Johannesburg", "Belo Horizonte", "Dalian", "Singapore", "Qingdao", "Zhengzhou", "Ji nan Shandong", "Abidjan", "Barcelona",
        "Yangon", "Addis Ababa", "Alexandria", "Saint Petersburg", "Nairobi", "Chittagong", "Guadalajara", "Fukuoka", "Ankara", "Hanoi",
        "Melbourne", "Monterrey", "Sydney", "Changsha", "Urumqi", "Cape Town", "Jiddah", "Brasilia", "Kunming", "Changchun",
        "Kabul", "Hefei", "Yaounde", "Ningbo", "Shantou", "New Taipei", "Tel Aviv", "Kano", "Shijiazhuang", "Montreal",
        "Rome", "Jaipur", "Recife", "Nanning", "Fortaleza", "Kozhikode", "Porto Alegre", "Taiyuan Shanxi", "Douala", "Ekurhuleni",
        "Malappuram", "Medellin", "Changzhou", "Kampala", "Antananarivo", "Lucknow", "Abuja", "Nanchang", "Wenzhou", "Xiamen",
        "Ibadan", "Fuzhou Fujian", "Salvador", "Casablanca", "Tangshan Hebei", "Kumasi", "Curitiba", "Bekasi", "Faisalabad", "Los Angeles",
        "Guiyang", "Port Harcourt", "Thrissur", "Santo Domingo", "Berlin", "Asuncion", "Dakar", "Kochi", "Wuxi", "Busan",
        "Campinas", "Mashhad", "Sanaa", "Puebla", "Indore", "Lanzhou", "Ouagadougou", "Kuwait City", "Lusaka", "Kanpur",
        "Durban", "Guayaquil", "Pyongyang", "Milan", "Guatemala City", "Athens", "Depok", "Izmir", "Nagpur", "Surabaya",
        "Handan", "Coimbatore", "Huaian", "Port-au-Prince", "Zhongshan", "Dubai", "Bamako", "Mbuji-Mayi", "Kiev", "Lisbon",
        "Weifang", "Caracas", "Thiruvananthapuram", "Algiers", "Shizuoka", "Lubumbashi", "Cali", "Goiania", "Pretoria", "Shaoxing",
        "Incheon", "Yantai", "Zibo", "Huizhou", "Manchester", "Taipei", "Mogadishu", "Brazzaville", "Accra", "Bandung",
        "Damascus", "Birmingham", "Vancouver", "Toluca de Lerdo", "Luoyang", "Sapporo", "Chicago", "Tashkent", "Patna", "Bhopal",
        "Tangerang", "Nantong", "Brisbane", "Tunis", "Peshawar", "Medan", "Gujranwala", "Baku", "Hohhot", "San Juan",
        "Belem", "Rawalpindi", "Agra", "Manaus", "Kannur", "Beirut", "Maracaibo", "Liuzhou", "Visakhapatnam", "Baotou",
        "Vadodara", "Barranquilla", "Phnom Penh", "Sendai", "Taoyuan", "Xuzhou", "Houston", "Aleppo", "Tijuana", "Esfahan",
        "Nashik", "Vijayawada", "Amman", "Putian", "Multan", "Grande Vitoria", "Wuhu Anhui", "Mecca", "Kollam", "Naples",
        "Daegu", "Conakry", "Yangzhou", "Havana", "Taizhou Zhejiang", "Baoding", "Perth", "Brussels", "Linyi Shandong", "Bursa",
        "Rajkot", "Minsk", "Hiroshima", "Haikou", "Daqing", "Lome", "Lianyungang", "Yancheng Jiangsu", "Panama City", "Almaty",
        "Semarang", "Hyderabad", "Valencia", "Davao City", "Vienna", "Rabat", "Ludhiana", "Quito", "Benin City", "La Paz",
        "Baixada Santista", "West Yorkshire", "Can Tho", "Zhuhai", "Leon de los Aldamas", "Quanzhou", "Matola", "Datong", "Sharjah", "Madurai",
        "Raipur", "Adana", "Santa Cruz", "Palembang", "Mosul", "Cixi", "Meerut", "Gaziantep", "La Laguna", "Batam",
        "Turin", "Warsaw", "Jiangmen", "Varanasi", "Hamburg", "Montevideo", "Budapest", "Lyon", "Xiangyang", "Bucharest",
        "Yichang", "Yinchuan", "Shiraz", "Kananga", "Srinagar", "Monrovia", "Tiruppur", "Jamshedpur", "Suqian", "Aurangabad",
        "Qinhuangdao", "Stockholm", "Anshan", "Glasgow", "Xining", "Makassar", "Hengyang", "Novosibirsk", "Ulaanbaatar", "Onitsha",
        "Jilin", "Anyang", "Auckland", "Tabriz", "Muscat", "Calgary", "Phoenix", "Qiqihaer", "N-Djamena", "Marseille",
        "Cordoba", "Jodhpur", "Kathmandu", "Rosario", "Tegucigalpa", "Ciudad Juarez", "Harare", "Karaj", "Medina", "Jining Shandong",
        "Abu Dhabi", "Munich", "Ranchi", "Daejon", "Zhangjiakou", "Edmonton", "Mandalay", "Gaoxiong", "Kota", "Natal",
        "Nouakchott", "Jabalpur", "Huainan", "Grande Sao Luis", "Asansol", "Philadelphia", "Yekaterinburg", "Gwangju", "Yiwu", "Chaozhou",
        "San Antonio", "Gwalior", "Ganzhou", "Homs", "Niamey", "Mombasa", "Allahabad", "Basra", "Kisangani", "San Jose",
        "Amritsar", "Taizhou Jiangsu", "Chon Buri", "Jiaxing", "Weihai", "Hai Phong", "Ottawa", "Zurich", "Taian Shandong", "Queretaro",
        "Joao Pessoa", "Kaifeng", "Cochabamba", "Konya", "Liuyang", "Liuan", "Rizhao", "Kharkiv", "Dhanbad", "Nanchong",
        "Dongying", "Belgrade", "Zunyi", "Zhanjiang", "Bucaramanga", "Uyo", "Copenhagen", "San Diego", "Shiyan", "Taizhong",
        "Bareilly", "Pointe-Noire", "Adelaide", "Suweon", "Mwanza", "Mianyang Sichuan", "Samut Prakan", "Maceio", "Qom", "Antalya",
        "Joinville", "Tengzhou", "Yingkou", "Ad-Dammam", "Suzhou", "Tanger", "Freetown", "Helsinki", "Aligarh", "Moradabad",
        "Pekan Baru", "Maoming", "Lilongwe", "Porto", "Prague", "Astana", "Jieyang", "Fushun Liaoning", "Mysore", "Abomey-Calavi",
        "Ruian", "Fes", "Port Elizabeth", "Florianopolis", "Ahvaz", "Bukavu", "Dallas", "Nnewi", "Kazan", "Jinhua",
        "San Luis Potosi", "Baoji", "Durg-Bhilainagar", "Bhubaneswar", "Kigali", "Sofia", "Pingdingshan Henan", "Dublin", "Puning", "Chifeng",
        "Zhuzhou", "Bujumbura", "Zhenjiang Jiangsu", "Liupanshui", "Barquisimeto", "Islamabad", "Huaibei", "Tasikmalaya", "Maracay", "Bogor",
        "Da Nang", "Nanyang Henan", "Nizhniy Novgorod", "Xiangtan Hunan", "Pizhou", "Tiruchirappalli", "Chelyabinsk", "Mendoza", "Luohe", "Xiongan",
        "Chandigarh", "Merida", "Jinzhou", "Benxi", "Binzhou", "Aba", "Chiang Mai", "Bazhong", "Quetta", "Kaduna",
        "Guilin", "Saharanpur", "Hubli-Dharwad", "Yueqing", "Guwahati", "Mexicali", "Salem", "Maputo", "Tripoli", "Haifa",
        "Bandar Lampung", "Bobo-Dioulasso", "Amsterdam", "Shimkent", "Omsk", "Aguascalientes", "Hargeysa", "Krasnoyarsk", "Xinxiang", "Siliguri", 
        "Samara", "Zaozhuang"
    ]


    G, edges = create_country_graph(countries)
    
    print("Training unsupervised node2vec model.")
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42) # edge masking control here
    
    n2v_model = UnsupervisedNode2Vec(G)
    n2v_model.create_training_graph(train_edges)
    n2v_model.train_embeddings()
    
    n2v_scores = [n2v_model.get_edge_score(edge) for edge in test_edges]
    print(f"Node2Vec scores on masked edges - Mean: {np.mean(n2v_scores):.4f}, Std: {np.std(n2v_scores):.4f}")
    

    print("\nTraining unsupervised GAE model.")
    node_features = np.array([n2v_model.embeddings[node] for node in countries])
    
    pyg_data, node_to_idx, idx_to_node = prepare_masked_pyg_data(G, node_features, countries)

    in_channels = node_features.shape[1]
    encoder = GNNEncoder(in_channels=in_channels, hidden_channels=64, out_channels=32)
    gae_model = UnsupervisedGAE(encoder)
    
    optimizer = torch.optim.Adam(gae_model.parameters(), lr=0.01)
    gae_model = train_gae(gae_model, pyg_data, optimizer)

    gae_model.eval()
    with torch.no_grad():
        z = gae_model.encode(pyg_data.x, pyg_data.train_pos_edge_index)
        auc, ap = gae_model.test(z, pyg_data.test_pos_edge_index, pyg_data.test_neg_edge_index)
        print(f"\nFinal GAE Test Results:")
        print(f"AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")

if __name__ == "__main__":
    main()