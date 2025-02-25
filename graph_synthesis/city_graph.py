import networkx as nx
from pyvis.network import Network

cities = [
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
    "Bandar Lampung", "Bobo-Dioulasso", "Amsterdam", "Shimkent", "Omsk", "Aguascalientes", "Hargeysa", "Krasnoyarsk", "Xinxiang", "Siliguri", "Samara", "Zaozhuang"

]

G = nx.DiGraph(directed=True)
edges = []
for i in range(len(cities)):
    last_letter = cities[i][-1].upper() 
    for j in range(len(cities)):
        if i != j:
            first_letter = cities[j][0].upper()  
            if last_letter == first_letter:
                edges.append([cities[i], cities[j]])
for edge in edges:
    G.add_edge(edge[0], edge[1])

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

nx.write_gexf(G, "city_graph.gexf")
