import numpy as np # pour les matrice
import random #des nbr aleatoire 
import math 

from termcolor import colored #pour la couleur du texte
import pyfiglet #pour les titres

import colorama
colorama.init()

# pour les titres
titre = pyfiglet.figlet_format("CHIO Algorithm")
print(colored(titre, color="cyan"))


#function parameter

C=0.18
PM1=0.22
R=0.25
H=0.10
T=0.08
V=0.12
PM2=0.18
PR=0.05

# Define the functions for each air quality parameter
# These functions calculate the fitness value based on the air quality parameters.
def g(co2):
    if co2<=1000:
        return co2/800
    else:
        return math.exp(0.002*(co2-1000))
        
h =lambda pm25: 0.1*pm25+0.01*(pm25)**1.7

k = lambda Radon: math.log(1+0.01*Radon)*Radon/50

def m (H):
    if H<40 or H>60:
       return 2*abs(H-50)**0.8
    else :
       return 0.5*abs(H-50)

def n(t):
    if t<18 :
        return 0.2*(20-t)**2
    elif t>18 and t<22 :
        return 0.1*abs(t-22)
    elif t>24 :
        return 0.3*(t-25)**1.5
    else:
        return 0

p=lambda voc,t: voc*(1+0.02*abs(t-23))/300

q=lambda pm1,pm25: pm1/8+0.05*pm1*math.sqrt(pm25)

r = lambda pres: 1/(1+math.exp(-0.5*(abs(pres-1013)-15)))


#initialization
# This function calculates the objective function for a room based on its air quality parameters.
def objective_function(room):

    fitness = (C * g(room[3]) + PM2 * h(room[2]) + R * k(room[0]) + 
               H * m(room[5]) + T * n(room[6]) + V * p(room[4], room[6]) + 
               PM1 * q(room[1], room[2]) + PR * r(room[7]))
    return fitness
      

# Columns: Radon, PM1, PM2.5, CO2, VOC, Humidity, Temp ,pression
     
Rooms = np.array([
        [15, 25, 26, 497, 159, 62, 22, 1008],
        [15, 13, 16, 682, 603, 65, 21, 998],
        [6,  18, 18, 535, 160, 54, 22, 998],
        [12, 32, 33, 501, 149, 56, 23, 1008]
        
    ])

# recommendations based on air quality parameters
# This function provides recommendations based on the air quality parameters of a room.# توصيات بناءً على معايير جودة الهواء
def air_quality_recommendations(Room):
    
    if Room[0] < 10:
        print(colored("  Radon level is below natural levels (unlikely).",color="red"))
    elif Room[0] >= 100:
        print(colored("  Radon level is too high. ",color="red"))
        print(colored("Risk:", color="blue"), "Lung cancer (+16%/100 Bq/m³).")
        print(colored("Solution:", color="blue"), "Sub-slab depressurization system (+5 kWh/year).")

    if Room[1] < 5:
        print(colored("  PM1 level is too low (filtration cost likely excessive.",color="red"))
        print(colored("Solution:",color="blue"), "Use cyclic filtration to save 15% energy.")
    elif Room[1] > 30:
        print(colored("  PM1 level is too high.", color="red"))
        print(colored("Risk:",color="blue"), "Alveolar penetration, inflammation.")
        print(colored("Solution:",color="blue"), "HEPA purifiers + source control (e.g., cooking).")

    if Room[2] < 5:
        print(colored("  PM2.5 level is too low (possibly due to unnecessary HEPA use).", color="red"))
        print(colored("Solution:", color="blue"), "Disable filtration if outdoor level < 3 µg/m³.")
    elif Room[2] > 25:
        print(colored("  PM2.5 level is too high.", color="red"))
        print(colored("Risk:", color="blue"), "Cardiopulmonary diseases, lung cancer.")
        print(colored("Solution:", color="blue"), "Use MERV-13 HEPA filters + source control.")

    if Room[3] < 400:
        print(colored("  CO₂ level is abnormally low (overventilation).", color="red"))
        print(colored("Risk:", color="blue"), "Respiratory discomfort, energy waste (+40%).")
        print(colored("Solution:", color="blue"), "Reduce ventilation rate with monitoring.")
    elif Room[3] > 1000:
        print(colored("  CO₂ level is too high.", color="red"))
        print(colored("Risk:", color="blue"), "Cognitive decline, headaches.")
        print(colored("Solution:", color="blue"), "Use mechanical ventilation with heat recovery (HRV).")

    if Room[4] < 50:
        print(colored("  VOC level is unusually low (possibly zero-emission materials).", color="red"))
    elif Room[4] > 500:
        print(colored("  VOC level is too high. ", color="red"))
        print(colored("Risk:", color="blue"), "Neurological irritation, cancer.")
        print(colored("Solution:", color="blue"), "Use activated carbon purifiers + low-VOC materials.")

    if Room[5] < 40:
        print(colored("  Humidity is too low",color="red"))
        print(colored("Risk:", color="blue"), "Eye dryness, virus propagation.")
        print(colored("Solution:", color="blue"), "Use cold-evaporation humidifiers (energy efficient).")
    elif Room[5] > 60:
        print(colored("  Humidity is too high.", color="red"))
        print(colored("Risk:", color="blue"), "Mold, dust mites.")
        print(colored("Solution:", color="blue"), "Use silent compressor dehumidifiers.")

    if Room[6] < 20:
        print(colored("  Temperature is too low.", color="red")) 
        print(colored("Risk:", color="blue"), "Hypothermia, insulation issues.")
        print(colored("Solution:", color="blue"), "Use low-temperature radiant heating.")
    elif Room[6] > 24:
        print(colored("  Temperature is too high.", color="red"))
        print(colored("Risk:", color="blue"), "Heat stroke, overuse of AC.")
        print(colored("Solution:", color="blue"), "Use solar or geothermal air conditioning.")

    if Room[7] < 980:
        print(colored("  Air pressure is too low.",color="red"))
        print(colored("Risk:", color="blue"), "Pollutant infiltration, discomfort.")
        print(colored("Solution:", color="blue"), "Use gentle pressurization systems.")
    elif Room[7] > 1040:
        print(colored("  Air pressure is too high",color="red"))
        print(colored("Risk:", color="blue"), "Ear discomfort, fatigue.")
        print(colored("Solution:", color="blue"), "Use automatic pressure regulators.")

# Function to print room parameters
def print_room(room):
    print("="*25)

    print("● " + colored("Radon: ", color="green") + f"{room[0]:.2f} Bq/m³")
    print("● " + colored("PM1: ", color="green") +  f"{room[1]:.2f} µg/m³")
    print("● " + colored("PM2.5: ", color="green") +f"PM2.5: {room[2]:.2f} µg/m³")
    print("● " + colored("CO2: ", color="green") +f"CO2: {room[3]:.2f} ppm")
    print("● " + colored("VOC: ", color="green") +f"VOC: {room[4]:.2f} ppb")
    print("● " + colored("Humidity: ", color="green") +f"Humidity: {room[5]:.2f}%")
    print("● " + colored("Temp: ", color="green") +f"Temperature: {room[6]:.2f}°C")
    print("● " + colored("Pressure: ", color="green") +f"Pressure: {room[7]:.2f} hPa")
   
  


#dictionnaire qui définit les valeurs minimales et maximales pour chaque paramètre d’air (Radon, PM1, etc.).

PARAM_BOUNDS = {
    'Radon': [10, 100] ,        # Bq/m³
    'PM1': [5, 30],             # µg/m³0
    'PM2.5': [5, 25],           # µg/m³
    'CO2': [400, 1000],         # ppm
    'VOC': [50, 500],           # ppb
    'Humidity': [40, 60],       # %
    'Temp': [20, 24],           # °C
    'Pressure': [980, 1040]     # hPa
}

#generate population

def generate_random_room():
    # """Generate a new random room within parameter bounds"""
    return np.array([
        random.uniform(PARAM_BOUNDS['Radon'][0], PARAM_BOUNDS['Radon'][1]),     # Radon
        random.uniform(PARAM_BOUNDS['PM1'][0], PARAM_BOUNDS['PM1'][1]),          # PM1
        random.uniform(PARAM_BOUNDS['PM2.5'][0], PARAM_BOUNDS['PM2.5'][1]),      # PM2.5
        random.uniform(PARAM_BOUNDS['CO2'][0], PARAM_BOUNDS['CO2'][1]),          # CO2
        random.uniform(PARAM_BOUNDS['VOC'][0], PARAM_BOUNDS['VOC'][1]),          # VOC
        random.uniform(PARAM_BOUNDS['Humidity'][0], PARAM_BOUNDS['Humidity'][1]),# Humidity
        random.uniform(PARAM_BOUNDS['Temp'][0], PARAM_BOUNDS['Temp'][1]),        # Temp
        random.uniform(PARAM_BOUNDS['Pressure'][0], PARAM_BOUNDS['Pressure'][1]) # Pressure
    ])

# This function implements the Chio algorithm to optimize room parameters based on air quality.
def Chio(Rooms,MAX_ITERATION=500,BRR=0.07,MAX_AGE=100):
   
    # step3 de chio:
    # Initialize parameters
    # This function initializes the parameters for the Chio algorithm, including population size, dimensions,
    population_size=len(Rooms)
    DIM=len(Rooms[0])

    INITIAL_INFECTED=1
    
    Stat=[0 for _ in range(population_size)]
    Age=[0 for _ in range(population_size)]
    is_corona = [False for _ in range(population_size)]
    
  # Randomly infect some individuals
  # This part randomly selects individuals to be infected at the start of the algorithm.
    for i in random.sample(range(population_size), INITIAL_INFECTED):
        Stat[i] = 1
    
    
    # This part initializes the age of each individual in the population.
    # Age is set to 0 for all individuals.
    for t in range(MAX_ITERATION):
        
        infected_indices = []
        susceptible_indices = []
        immunized_indices = []
        
        fitness_values = [objective_function(Rooms[i]) for i in range(population_size)]
        mean_fitness = sum(fitness_values) / population_size
        new_person = generate_random_room()
        fitness_new = objective_function(new_person)
    # This part categorizes individuals based on their infection status.
        # It creates lists for infected, susceptible, and immunized individuals.
        for i in range(len(Stat)):
          if Stat[i] == 1:
            infected_indices.append(i)
          elif Stat[i] == 0:
            susceptible_indices.append(i)
          elif Stat[i] == 2:
            immunized_indices.append(i)
    # This part prints the current iteration number and the number of infected individuals.
        for i in range (population_size):
            proch=Rooms[i].copy()
            for j in range(DIM):
                r_val = random.uniform(0,1)
                if r_val < 1/3*BRR and infected_indices: # If the random number is less than 1/3 of BRR and there are infected individuals
                    proch[j] = Rooms[i][j] + r_val * (Rooms[i][j] - Rooms[random.choice(infected_indices)][j])
                    is_corona[i] = True # Mark as infected
                elif r_val < 2/3*BRR and susceptible_indices: # If the random number is less than 2/3 of BRR and there are susceptible individuals
                    proch[j] = Rooms[i][j] + r_val * (Rooms[i][j] - Rooms[random.choice(susceptible_indices)][j])
                elif r_val < BRR and immunized_indices: # If the random number is less than BRR and there are immunized individuals
                    proch[j] = Rooms[i][j] + r_val * (Rooms[i][j] - Rooms[random.choice(immunized_indices)][j])
       # Ensure parameters stay within bounds 
       # np.clip(x, min, max)
                proch = np.clip(proch, 
                                  [PARAM_BOUNDS['Radon'][0], PARAM_BOUNDS['PM1'][0], PARAM_BOUNDS['PM2.5'][0],
                                  PARAM_BOUNDS['CO2'][0], PARAM_BOUNDS['VOC'][0], PARAM_BOUNDS['Humidity'][0],
                                  PARAM_BOUNDS['Temp'][0], PARAM_BOUNDS['Pressure'][0]],
                                  [PARAM_BOUNDS['Radon'][1], PARAM_BOUNDS['PM1'][1], PARAM_BOUNDS['PM2.5'][1],
                                  PARAM_BOUNDS['CO2'][1], PARAM_BOUNDS['VOC'][1], PARAM_BOUNDS['Humidity'][1],
                                  PARAM_BOUNDS['Temp'][1], PARAM_BOUNDS['Pressure'][1]])
                

                
    #step4 de chio:
            # Evaluate the new room
            if objective_function(proch)<fitness_values[i]:
              Rooms[i]=proch
            else:
              Age[i]+=1           
        
            if objective_function(proch)< (fitness_new/mean_fitness) and Stat[i]==0 and is_corona[i]:
                Stat[i]=1
                Age[i]=1
                
            if objective_function(proch)> (fitness_new/mean_fitness) and Stat[i]==1 :
                Stat[i]=2
                Age[i]=0
    
    #step5 de chio: 

            if Age[i]>=MAX_AGE and Stat[i]==1 :
                Rooms[i] = generate_random_room()
                Stat[i]=0
                Age[i]=0
         # This part finds the optimal room based on the objective function.       
        Optimal_idx = np.argmin([objective_function(room) for room in Rooms])
        return Rooms[Optimal_idx],objective_function(Rooms[Optimal_idx])
      

# Main execution to evaluate rooms and print results
# This part evaluates each room's air quality parameters and prints the fitness value.
fitList = []

for i in range(len(Rooms)): 
    print(colored(f"\nRoom {i+1} parameters:",color="green"))
    print_room(Rooms[i])
    fit=objective_function(Rooms[i])
    print("●" + f"Fitness value: {fit:.4f}")
    fitList.append(fit)
    air_quality_recommendations(Rooms[i])
best_idx = np.argmin(fitList)
best_room = Rooms[best_idx].copy() 

Optimal_room, best_fitness = Chio(Rooms)

while(best_fitness>0.6):
       Optimal_room, best_fitness = Chio(Rooms,500,random.uniform(0.05, 0.07),random.randint(50, 150))
    
print(colored("\nOptimal room parameters:", color="green"))
print_room(Optimal_room)
print(colored(f"\nFitness value: {best_fitness:.4f}", color="grey"))

print("="*50,end="")
print(colored(f"\nBest room is Room {best_idx + 1} with fitness value:{fitList[best_idx]:.4f}", color="grey"))
 #This function provides recommendations based on the best room's parameters. 
print("="*50)
