import os

import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import random
import torch
import torch.nn.functional as F
from infomap import Infomap

random.seed(42)
np.random.seed(42)

#Local Methods


#To simulate extinction, we will set a percent threshold X for the species to be removed.
#If less than X percent of the in-links for a species remain, the species will be secondarily removed.
#This will be done iteratively until no more species can be removed.
#The percent threshold will be set to 0.25 for now.

#Define a function that returns a new graph with the species removed, along with secondary extinctions
def remove_species(g, species_to_remove, threshold=0.5):

    # Create a copy of the graph and list to avoid modifying the original
    species_to_remove = list(species_to_remove)

    g_copy = g.copy()

    if species_to_remove:
        #Find basal species, exclude them from being secondarily extinguished
            #Otherwise they will be popped by having no in-links, and will be confused with low-tier species
        #Put before primary removals so that low-tier species are not confused with basal species
        basal_species = []
        for i in range(0, g_copy.vcount()):
            if g_copy.degree(i, mode = "in") == 0:
                basal_species.append(g_copy.vs[i]["name"]) #need to work by name due to dynamic nature of the graph

        #Remove specified species
        g_copy.delete_vertices(species_to_remove)

        #Clear array of species to remove
        species_to_remove = []
        
        #Iterate through the graph until no more species can be removed
        while True:
            for i in range(0, g_copy.vcount()):
                
                if g_copy.vs[i]["name"] not in basal_species:
                    if (g_copy.degree(i, mode = "in")) / g.vs.find(name = g_copy.vs[i]["name"]).degree(mode = "in") < (threshold + 0.000001):                  
                        
                        species_to_remove.append(i)
            
            if species_to_remove:
                g_copy.delete_vertices(species_to_remove)
            else:
                break
            species_to_remove = [] #Clear array of species to remove, otherwise will try popping missing indices twice

    return g_copy



def extinction_simulation(g, type_ = "top", metric = "ksi", percent = 10, species_to_keep = [], threshold=0.5):

    metrics = ["ksi", "trophic_level"]
    types = ["top", "median", "bottom"]

    if metric not in metrics:
        raise ValueError("Metric must be one of 'ksi' or 'trophic_level'")
    if type_ not in types:
        raise ValueError("Type must be one of 'top', 'median', or 'bottom'")
    
    if metric == "ksi":
        # Get the KSI values for each species
        ksi = ksi_indexer(g)
        
        # Get the indices of the top X percent of species in the list ksi
        species_to_remove = splicer(ksi, percent, type_)
    elif metric == "trophic_level":
        # Get the trophic levels for each species
        trophic_levels = list(trophic_level_indexer(g))
        
        # Get the indices of the top X percent of species in the list trophic_levels
        species_to_remove = splicer(trophic_levels, percent, type_, degrees=g.vs.degree())

    if species_to_keep:
        for i in species_to_keep:
            species_to_remove.pop(species_to_remove.index(i))


    # Remove the specified species and return the modified graph
    #Remove species already copies the graph no need to copy again
    return remove_species(g, species_to_remove, threshold), g.vs[species_to_remove]["name"]






def infomap_flow_indexer(g):
    """Assign a flow value to each vertex based on a random walk.

    The function uses the Infomap algorithm to compute the flow (stationary
    distribution of a random walker). The resulting probabilities are stored in
    the vertex attribute ``flow`` and returned as a list.
    """
    # Ensure each node has a unique name (Infomap can use string IDs directly)

    #break condition for if there are no connections left, i.e. only basal species left
    if (len(g.es) < 1):
        g.vs["flow"] = [0.0] * g.vcount()
        return g.vs["flow"]

    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(i) for i in range(g.vcount())]

    # Create Infomap instance
    im = Infomap("--two-level")

    # Add edges to Infomap (no weights assumed)
    for edge in g.es:
        source = g.vs[edge.source].index
        target = g.vs[edge.target].index
        im.add_link(source, target)

    # Run the Infomap clustering
    im.run()

    # Initialize all flows to 0
    flow_dict = {v.index: 0.0 for v in g.vs}

    # Map flow values using actual string node IDs
    for node in im.nodes:
        flow_dict[node.node_id] = node.flow

    # Assign flows back to igraph vertices
    g.vs["flow"] = [flow_dict[v.index] for v in g.vs]

    return g.vs["flow"]

#sets the KSI values for each species; internal methods can be changed to add more metrics
#currently uses degree, flow, and betweenness
def ksi_indexer(g):

    ksi = [0] * g.vcount()

    flow = infomap_flow_indexer(g)
    methods = [g.vs.degree, lambda: flow, g.vs.betweenness] #

    #Find basal and apex indices
    basal_indices = [i.index for i in g.vs if g.degree(i.index, mode = "in") == 0 and g.degree(i.index, mode = "out") > 0]
    apex_indices = [i.index for i in g.vs if g.degree(i.index, mode = "in") > 0 and g.degree(i.index, mode = "out") == 0]


    for i in methods:
        temp = i()

        if i == g.vs.betweenness:
            avg_betweenness = np.mean(temp) #get the average betweenness of the network
            for j in basal_indices + apex_indices:
                temp[j] = avg_betweenness
            #Award each basal and apex species the avg betweenness of the ENTIRE network, not just those with nonzero betweeness
            #This is to avoid having basal and apex species with 0 betweenness, which would skew the KSI index
            #But, it also doesn't give them the average betweeness of the middle trophic levels, as this would be too high


        zipped_list = zip(list(range(0, len(temp))), temp)
        rankings = sorted(zipped_list, key = lambda x: x[1], reverse = True) #higher metrics at lower indices, that's why it's reversed
        for j in range(len(rankings)):
            ksi[rankings[j][0]] += rankings[j][1]

    ksi = [ksi[i] / (len(methods) * g.vcount()) for i in range(len(ksi))] ##normalizes the KSI index to be between 0 and 1
    g.vs["ksi"] = ksi
    return ksi

#Straight from copilot, but it works so why not
def trophic_level_indexer(g):
    assert g.is_directed(), "Graph must be directed!"

    n = g.vcount()
    A = np.array(g.get_adjacency().data)  # Adjacency matrix
    A = A.T  # Transpose: edge from prey to predator

    # Initialize trophic levels with NaN (to identify unprocessed nodes)
    TL = np.full(n, np.nan)

    # Handle disconnected basal species (no in-links and no out-links)
    disconnected_basal = [i for i in range(n) if g.degree(i, mode="in") == 0 and g.degree(i, mode="out") == 0]
    for i in disconnected_basal:
        TL[i] = 1  # Assign trophic level of 1 to disconnected basal species

    # Identify weakly connected components (sub-webs)
    components = g.components(mode="weak")

    for component in components:
        if len(component) > 1:  # Only compute trophic levels for components with more than one node
            subgraph = g.subgraph(component)  # Extract the sub-web
            sub_n = subgraph.vcount()

            # Adjacency matrix for the sub-web
            sub_A = np.array(subgraph.get_adjacency().data).T

            # Handle nodes with 0 in-degree separately
            in_degrees = np.array(subgraph.degree(mode="in"))
            D_inv = np.zeros((sub_n, sub_n))
            for i in range(sub_n):
                if in_degrees[i] > 0:
                    D_inv[i, i] = 1.0 / in_degrees[i]

            # Solve the linear system for the sub-web
            M = np.eye(sub_n) - np.dot(D_inv, sub_A)
            b = np.ones(sub_n)

            # Add regularization to avoid singular matrix errors
            epsilon = 1e-10
            M += np.eye(sub_n) * epsilon

            # Solve for trophic levels in the sub-web
            sub_TL = np.round(np.linalg.solve(M, b), decimals=9)

            # Assign the computed trophic levels back to the original graph
            for i, node in enumerate(component):
                TL[node] = sub_TL[i]
        else:
            # For single-node components, assign a trophic level of 1
            TL[component[0]] = 1

    # Assign the computed trophic levels to the graph
    g.vs["trophic_level"] = list(TL)

    return TL



#returns a list of indices of the top X percent of species in the list ksi
#lower = True will return the bottom X percent of species instead, but still in decreasing order of KSI
def splicer(vals, percent_amount, tier="top", degrees=[]):
    # Tier options: "top", "bottom", "median"
    if tier not in ["top", "bottom", "median"]:
        raise ValueError("Tier must be one of 'top', 'bottom', or 'median'")

    vals = [round(v, 9) for v in vals]  # Round to avoid floating-point issues
    percent_amount = percent_amount / 100
    number_of_species = int(len(vals) * percent_amount + 0.00001)

    pairs = list(enumerate(vals))  # Pair index with value
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)  # Stable sort

    output = []

    if not vals:  # Handle empty input
        return output

    if percent_amount > 0:  # Fixed amount, no tie-breaking
        if tier == "bottom":
            output.extend(x[0] for x in sorted_pairs[-number_of_species:])
        elif tier == "top":
            output.extend(x[0] for x in sorted_pairs[:number_of_species])
        elif tier == "median":
            output.extend(x[0] for x in sorted_pairs[int(len(vals) / 2 - number_of_species / 2):int(len(vals) / 2 + number_of_species / 2)])
    else:  # Single index, with tie-breaking
        if not degrees or len(sorted_pairs) < 2:
            if tier == "bottom":
                output.append(sorted_pairs[-1][0])
            elif tier == "top":
                output.append(sorted_pairs[0][0])
            elif tier == "median":
                output.append(sorted_pairs[int(len(vals) / 2)][0])
        elif degrees:
            if tier == "bottom":
                temp = sorted_pairs[-1][1]
                index = -1
                while sorted_pairs[index][1] == temp and index > -len(sorted_pairs):
                    index -= 1
                index += 1
                max_index = max(range(len(sorted_pairs) + index, len(sorted_pairs)), key=lambda i: degrees[sorted_pairs[i][0]])
                output.append(sorted_pairs[max_index][0])
            elif tier == "top":
                temp = sorted_pairs[0][1]
                index = 0
                while sorted_pairs[index][1] == temp and index < len(sorted_pairs) - 1:
                    index += 1
                index -= 1
                max_index = max(range(0, index + 1), key=lambda i: degrees[sorted_pairs[i][0]])
                output.append(sorted_pairs[max_index][0])
            elif tier == "median":
                median_value = sorted_pairs[int(len(vals) / 2)][1] #np.median(vals)
                tolerance = 1e-13
                median_group = [i for i, v in sorted_pairs if abs(v - median_value) < tolerance]
                if median_group:
                    max_index = max(median_group, key=lambda i: (degrees[i], -i))
                    output.append(max_index)
                else: #problem here is that it doesn't even check degree
                    print("whoopsie")
                    closest_index = min(range(len(vals)), key=lambda i: (abs(vals[i] - median_value), i))
                    output.append(closest_index)
    return output


def role_identifier(g, connectivity_threshold = 0.62, within_module_threshold = 2.5):
    #Goal: find the role of each species in the network by considering modularity and connectivity
    #This will be done by finding the number of connections to each species, and the number of connections to other species in the same module

    #type_s of roles:
    #Network connector: high connectivity within and outside module
    #Module connector: few links, mostly between modules
    #Module Specialist: few links, mostly within module
    #Module Hub: high number of links within own modules

    #first, need to find the modules in the network

    #Initially, use eigenvector method to find the modules in the network; NOT stochastic


    community = g.community_leading_eigenvector() #find the modules in the network
    print("Number of modules:", len(set(community.membership)))


    #Get the membership of each vertex in the community
    membership = community.membership
    g.vs["module"] = membership

    set_z_c_roles(g, connectivity_threshold, within_module_threshold) #set the z and c roles for each species

    return community #returning the community to avoid re-running modularity func


def set_z_c_roles(g, connectivity_threshold = 0.62, within_module_threshold = 2.5):      

    ####PREPARING FOR Z SCORES#####
    mean_links_per_module = [0] * (max(g.vs["module"]) + 1)
    degree_std_per_module = [0] * (max(g.vs["module"]) + 1)

    for i in range(max(g.vs["module"]) + 1):
        temp = [j for j in range(0, g.vcount()) if g.vs[j]["module"] == i] #find all the vertices in the module
        
        temp_sums = [0] * len(temp) #array for the number of links for each node in the module

        for k in range(len(temp)):
            #Need to find the number of links within the module for each node of the module
            temp_sum = sum(1 for j in g.neighbors(temp[k], mode="all") if g.vs[temp[k]]["module"] == g.vs[j]["module"] and temp[k] != j)

            mean_links_per_module[i] += temp_sum #sum the number of links for each node in the module
            
            temp_sums[k] = temp_sum #store the number of links for each node in the module
            
            #This is then summed together and divided by the number of nodes in the module to get the mean number of links per module
        if len(temp) > 0:
            mean_links_per_module[i] = mean_links_per_module[i] / len(temp)
        else:
            mean_links_per_module[i] = 0
        
        #Now we need to find the standard deviation of the number of links for each node in the module

        for k in range(len(temp)):
            #Need to find the number of links within the module for each node of the module

            degree_std_per_module[i] += (temp_sums[k] - mean_links_per_module[i]) ** 2
        
        if len(temp) > 0:
            degree_std_per_module[i] = (degree_std_per_module[i] / len(temp)) ** 0.5
        else:
            degree_std_per_module[i] = 0

    
    for i in range(0, g.vcount()):

        #Z Score
    
        #Need to find the within-module degree
        g.vs[i]["within_module_degree"] = sum(1 for j in g.neighbors(i, mode="all") if g.vs[i]["module"] == g.vs[j]["module"] and i != j)
        
        #Now we need to find the z-score for the within-module degree
        #This is done by subtracting the mean number of links per module from the number of links for the node, and dividing by the standard deviation of the number of links for the module
        if degree_std_per_module[g.vs[i]["module"]] != 0:
            g.vs[i]["within_module_degree"] = (g.vs[i]["within_module_degree"] - mean_links_per_module[g.vs[i]["module"]]) / degree_std_per_module[g.vs[i]["module"]]
        else:
            g.vs[i]["within_module_degree"] = 0
        

        #C Score

        #Now need to find among module connectivity, quantified with the metric "Participation Coefficient"
        g.vs[i]["among_module_connectivity"] = 0

        for k in range(max(g.vs["module"]) + 1):
            current_module_degree = sum(1 for j in g.neighbors(i, mode="all") if g.vs[j]["module"] == k and i != j) #need ot make sure that integer k can be compared as such
            g.vs[i]["among_module_connectivity"] += (current_module_degree / g.degree(i, mode = "all")) ** 2

        g.vs[i]["among_module_connectivity"] = 1 - g.vs[i]["among_module_connectivity"] #Participation coefficient is 1 - sum of squares of the fraction of edges in each module


        #Roling

        #Now we can classify the species based on the metrics
        if g.vs[i]["among_module_connectivity"] > connectivity_threshold and g.vs[i]["within_module_degree"] > within_module_threshold:
            g.vs[i]["role"] = "Network Connector"
        elif g.vs[i]["among_module_connectivity"] > connectivity_threshold and g.vs[i]["within_module_degree"] < within_module_threshold:
            g.vs[i]["role"] = "Module Connector"
        elif g.vs[i]["among_module_connectivity"] < connectivity_threshold and g.vs[i]["within_module_degree"] < within_module_threshold:
            g.vs[i]["role"] = "Module Specialist"
        elif g.vs[i]["among_module_connectivity"] < connectivity_threshold and g.vs[i]["within_module_degree"] > within_module_threshold:
            g.vs[i]["role"] = "Module Hub"

def compare_and_color_modules(g, name=None):
    # Run the modularity methods
    leading_eigenvector_community = g.community_leading_eigenvector()
    spinglass_community = g.community_spinglass(update_rule="config")

    # Get module memberships
    leading_membership = leading_eigenvector_community.membership
    spinglass_membership = spinglass_community.membership

    # Number of modules in each method
    num_leading_modules = max(leading_membership) + 1
    num_spinglass_modules = max(spinglass_membership) + 1

    # Create a mapping from spinglass modules to leading eigenvector modules
    module_mapping = {}
    for spinglass_module in range(num_spinglass_modules):
        # Find the leading eigenvector module with the most overlap
        overlap = [
            sum(
                1
                for i in range(len(leading_membership))
                if leading_membership[i] == leading_module
                and spinglass_membership[i] == spinglass_module
            )
            for leading_module in range(num_leading_modules)
        ]
        # Map spinglass module to the leading eigenvector module with the most overlap
        module_mapping[spinglass_module] = overlap.index(max(overlap))

    # Assign colors to the modules
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    g.vs["color_leading"] = [colors[leading_membership[i] % len(colors)] for i in range(len(leading_membership))]
    g.vs["color_spinglass"] = [
        colors[module_mapping[spinglass_membership[i]] % len(colors)] for i in range(len(spinglass_membership))
    ]

    # Plot the graphs
    layout = g.layout("fr")  # Use a force-directed layout
    plt.figure(figsize=(12, 6))

    # Plot leading eigenvector result
    g.vs["module"] = leading_membership
    set_z_c_roles(g)

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        np.array(g.vs["among_module_connectivity"]),
        np.array(g.vs["within_module_degree"]),
        s=(np.array(g.vs["trophic_level"]) ** 4),
        c=g.vs["color_leading"],
    )
    plt.axhline(y=2.5, color="red", linestyle="--", linewidth=1, label="Horizontal Line")
    plt.axvline(x=0.62, color="blue", linestyle="--", linewidth=1, label="Vertical Line")
    plt.ylabel("Within-Module Degree (Z)")
    plt.xlabel("Among-Module Connectivity (C)")
    plt.legend()
    if name:
        plt.title(f"Leading Eigenvector Modules: {name}")
    else:
        plt.title("Leading Eigenvector Modules")

    # Add legend for leading eigenvector modules
    legend_labels = [f"Module {i}" for i in range(num_leading_modules)]
    handles = [plt.Line2D([0], [0], marker="o", color=colors[i % len(colors)], linestyle="None") for i in range(num_leading_modules)]
    plt.legend(handles, legend_labels, title="Modules", loc="upper left")

    print(
        "Network connector species (by leading eigenvector for {name}): "
        + str([i["name"] for i in g.vs if i["role"] == "Network Connector"])
    )

    # Plot spinglass result
    g.vs["module"] = spinglass_membership
    set_z_c_roles(g)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        np.array(g.vs["among_module_connectivity"]),
        np.array(g.vs["within_module_degree"]),
        s=(np.array(g.vs["trophic_level"]) ** 4),
        c=g.vs["color_spinglass"],
    )
    plt.axhline(y=2.5, color="red", linestyle="--", linewidth=1, label="Horizontal Line")
    plt.axvline(x=0.62, color="blue", linestyle="--", linewidth=1, label="Vertical Line")
    plt.ylabel("Within-Module Degree (Z)")
    plt.xlabel("Among-Module Connectivity (C)")
    plt.legend()
    if name:
        plt.title(f"Spinglass Modules: {name}")
    else:
        plt.title("Spinglass Modules")

    # Add legend for spinglass modules
    legend_labels = [f"Module {i}" for i in range(num_spinglass_modules)]
    handles = [plt.Line2D([0], [0], marker="o", color=colors[module_mapping[i] % len(colors)], linestyle="None") for i in range(num_spinglass_modules)]
    plt.legend(handles, legend_labels, title="Modules", loc="upper left")

    print(
        "Network connector species (by spinglass for {name}): "
        + str([i["name"] for i in g.vs if i["role"] == "Network Connector"])
    )

    plt.show()

    







def cumulative_degree_distribution(g, plot = False):

    #Fetch degrees
    degrees = np.array(g.degree())

    #Calculate the degree distribution
    degree_values, counts = np.unique(degrees, return_counts=True)

    pk = counts / np.sum(counts)  # Probability of each degree

    pk_cumulative = np.array([np.sum(pk[degree_values >= k]) for k in degree_values])

    if plot == True:
        plt.figure()
        plt.plot(degree_values, pk_cumulative, marker='o', linestyle='-')
        plt.xlabel('Degree k')
        plt.ylabel('P(≥k)')
        plt.title('Cumulative Degree Distribution')
        plt.yscale('log')  # often useful
        plt.xscale('log')
        plt.grid(True)
        plt.show()

    return degree_values, pk_cumulative


def fit_CDD(g, plot = True):
    # Fit a power law to the cumulative degree distribution
    degree_values, pk_cumulative = cumulative_degree_distribution(g)

    n = len(degree_values)

    #Function options to fit to
    def power_law(k, gamma, C):
        return C * k**(-gamma)

    def exponential(k, lambd, C):
        return C * np.exp(-lambd * k)

    def lognormal(k, mu, sigma, C):
        return C * (1/(k*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(k) - mu)**2/(2*sigma**2))

    #Goodness of fit functions
    def compute_rss(y_obs, y_pred):
        return np.sum((y_obs - y_pred) ** 2)

    def compute_aic(rss, n, p):
        return n * np.log(rss / n) + 2 * p

    def compute_bic(rss, n, p):
        return n * np.log(rss / n) + p * np.log(n)



    models = {
    "powerlaw": (power_law, [2.5, 1.0]),
    "exponential": (exponential, [0.1, 1.0]),
    "lognormal": (lognormal, [1.0, 1.0, 1.0])
    }

    results = {}

    for name, (model_func, p0) in models.items():
        try:
            popt, _ = curve_fit(model_func, degree_values, pk_cumulative, p0=p0)
            rss = compute_rss(pk_cumulative, model_func(degree_values, *popt))
            aic = compute_aic(rss, n, len(popt))
            bic = compute_bic(rss, n, len(popt))
            results[name] = {"params": popt, "rss": rss, "aic": aic, "bic": bic}
        except RuntimeError:
            print(f"Fit failed for {name}")
    

    sorted(results.items(), key=lambda x: x[1]['aic'])


    best_model = min(results.items(), key=lambda x: x[1]['aic'])

    if plot:

        plt.plot(degree_values, pk_cumulative, label="Original CDD", color="blue", marker="o", linestyle = "None")  # Line plot for y1
        
        for j in results.items():
            y_pred = models[j[0]][0](degree_values, *j[1]['params'])
            plt.plot(degree_values, y_pred, label=f"Fit: {j[0]}")  # Line plot for y2



        
        #plt.plot(degree_values, y_pred, label=f"Best Fit: {best_model[0]}", color="red")  # Line plot for y2
        #y_pred = models[best_model[0]][0](degree_values, *best_model[1]['params'])


        # Add labels, title, and legend
        plt.xlabel("Node Degree")
        plt.ylabel("Probability")
        plt.title("CDD Fitting")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()


    return best_model


#Method do incite extinction dynamically from the highest trophic level,
#and then see how many apex species are left, with what mean trophic level, and with what STD

def apex_leveling(g, name = None, plot = True, min_or_max = "max"):

        stds = []
        means = []
        num_apex = []
        num_basal = []
        min_trophic = []
        max_trophic = []
        total_species = []
        
        g_copy = g.copy()
        

        for i in range(g_copy.vcount()):
            #Get the trophic levels for each species
            trophic_levels = list(trophic_level_indexer(g_copy))
            if not trophic_levels:
                print("No trophic levels found. The graph might be empty or disconnected.")
                break  # Exit the loop if no trophic levels are found
            species_to_remove = []
            current_apex = [i for i in range(g_copy.vcount()) if g_copy.degree(i, mode = "in") > 0 and g_copy.degree(i, mode = "out") == 0]
            
            if not current_apex:
                #If theres no apex species left, make a list with only one element which has the highest trophic level
                #This is to avoid the case where there are no apex species left, and we need to find the one with the highest trophic level
                max_index = max(range(len(trophic_levels)), key=lambda i: trophic_levels[i])
                current_apex.append(max_index)#make it a list so we can iterate over it

            current_basal = [i for i in range(g_copy.vcount()) if g_copy.degree(i, mode = "in") == 0 and g_copy.degree(i, mode = "out") > 0]


            #Get the trophic levels for each species
            
            if current_apex:
                means.append(np.mean([trophic_levels[i] for i in current_apex]))
                stds.append(np.std([trophic_levels[i] for i in current_apex]))
                num_apex.append(len(current_apex))
                num_basal.append(len(current_basal))
                min_trophic.append(min([trophic_levels[i] for i in current_apex]))
                max_trophic.append(max([trophic_levels[i] for i in current_apex]))
                total_species.append(g_copy.vcount())
        
            #Get the indices of the top X percent of species in the list trophic_levels
            if min_or_max == "max":
                for i in current_apex:
                    if abs(trophic_levels[i] - max([trophic_levels[i] for i in current_apex])) < 1e-9:
                        species_to_remove.append(i)
                        break #break to avoid removing multiple apex species at once
            elif min_or_max == "min":
                for i in current_apex:
                    if abs(trophic_levels[i] - min([trophic_levels[i] for i in current_apex])) < 1e-9:
                        species_to_remove.append(i)
                        break



            #Remove the specified species and return the modified graph
            g_copy = remove_species(g_copy, species_to_remove)
            #Do again to remove species that have no in-links and no out-links
            species_to_remove = [i for i in range(g_copy.vcount()) if g_copy.degree(i, mode="in") == 0 and g_copy.degree(i, mode="out") == 0]
            g_copy = remove_species(g_copy, species_to_remove)

        


        if plot == True:
            plt.figure()
            plt.plot(range(g.vcount())[:len(means)], means, label="Mean Trophic Level of Apex", color="blue")
            plt.plot(range(g.vcount())[:len(stds)], stds, label="STD of Trophic Level of Apex", color="red")
            plt.plot(range(g.vcount())[:len(num_apex)], num_apex, label="Number of Apex Species", color="green")
            plt.plot(range(g.vcount())[:len(min_trophic)], min_trophic, label="Minimum Trophic Level of Apex", color="orange")
            plt.plot(range(g.vcount())[:len(max_trophic)], max_trophic, label="Maximum Trophic Level of Apex", color="purple")
            plt.plot(range(g.vcount())[:len(num_basal)], num_basal, label="Number of Basal Species", color="black")
            plt.plot(range(g.vcount())[:len(total_species)], total_species, label="Total Species", color="pink")
            plt.xlabel(f"Num of Original Species count Removed, dynamically from the {min_or_max} trophic level of apex")
            plt.legend()
            plt.ylabel("Metric Values")
            if not name:
                plt.title('Extinguishing Apex Species')
            else:
                plt.title(f'Extinguishing Apex Species: {name}')
            plt.grid(True)
            plt.show()
    
        return
        

def apex_pop(g, min_or_max = "max", iters = 1):

    g_copy = g.copy()

    for i in range(iters):
        #Get the trophic levels for each species
        trophic_levels = list(trophic_level_indexer(g_copy))
        species_to_remove = []
        current_apex = [i for i in range(g_copy.vcount()) if g_copy.degree(i, mode = "in") > 0 and g_copy.degree(i, mode = "out") == 0]
        
        if not current_apex:
            #If theres no apex species left, make a list with only one element which has the highest trophic level
            #This is to avoid the case where there are no apex species left, and we need to find the one with the highest trophic level
            max_index = max(range(len(trophic_levels)), key=lambda i: trophic_levels[i])
            current_apex.append(max_index)#make it a list so we can iterate over it

        #Get the indices of the top X percent of species in the list trophic_levels
        if min_or_max == "max":
            for i in current_apex:
                if abs(trophic_levels[i] - max([trophic_levels[i] for i in current_apex])) < 1e-9:
                    species_to_remove.append(i)
                    break #break to avoid removing multiple apex species at once
        elif min_or_max == "min":
            for i in current_apex:
                if abs(trophic_levels[i] - min([trophic_levels[i] for i in current_apex])) < 1e-9:
                    species_to_remove.append(i)
                    break

        #Remove the specified species and return the modified graph
        g_copy = remove_species(g_copy, species_to_remove)

        #Remove species that have no in-links and no out-links
        species_to_remove = [i for i in range(g_copy.vcount()) if g_copy.degree(i, mode="in") == 0 and g_copy.degree(i, mode="out") == 0]
        g_copy = remove_species(g_copy, species_to_remove)

    return g_copy


def plot_trophic_web_by_module(g, name=None):
    """
    Plots a trophic web with nodes positioned by their trophic levels and segmented into modules.
    Basal species and apex species are assigned specific colors, irrespective of their module.
    """
    # Compute trophic levels
    trophic_levels = trophic_level_indexer(g)

    # Determine modules using the leading eigenvector method
    walktrap_community = g.community_walktrap().as_clustering()
    walktrap_membership = walktrap_community.membership
    num_modules = max(walktrap_membership) + 1

    # Assign colors to modules
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    module_colors = [colors[walktrap_membership[i] % len(colors)] for i in range(len(walktrap_membership))]

    # Identify basal and apex species
    basal_species = [i for i in range(g.vcount()) if g.degree(i, mode="in") == 0 and g.degree(i, mode="out") > 0]
    apex_species = [i for i in range(g.vcount()) if g.degree(i, mode="in") > 0 and g.degree(i, mode="out") == 0]

    # Assign specific colors to basal and apex species
    basal_color = "gray"
    apex_color = "black"

    # Create a layout for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine x-coordinates for each module
    module_positions = np.linspace(0, 1, num_modules + 1)[:-1] + (1 / (2 * num_modules))

    # Plot nodes by module and trophic level
    node_positions = {}
    for module_index in range(num_modules):
        # Get nodes in the current module
        module_nodes = [i for i in range(len(walktrap_membership)) if walktrap_membership[i] == module_index]

        # Assign x-coordinates for nodes in this module with jitter
        x_pos = module_positions[module_index]
        for node in module_nodes:
            jittered_x_pos = x_pos + random.uniform(-0.02, 0.02)  # Add small jitter to x-coordinate
            y_pos = trophic_levels[node]  # Use trophic level for y-coordinate
            node_positions[node] = (jittered_x_pos, y_pos)

            # Determine color based on species type
            if node in basal_species:
                color = basal_color
            elif node in apex_species:
                color = apex_color
            else:
                color = module_colors[node]

            # Plot the node
            ax.scatter(jittered_x_pos, y_pos, color=color, s=50, label=f"Module {module_index}" if node == module_nodes[0] else "")

    # Plot edges
    for edge in g.es:
        source, target = edge.tuple
        if source in node_positions and target in node_positions:
            x_coords = [node_positions[source][0], node_positions[target][0]]
            y_coords = [node_positions[source][1], node_positions[target][1]]

            if (y_coords[1] - y_coords[0]) > 0:  # Positive slope
                ax.plot(x_coords, y_coords, color="gray", alpha=0.5, linewidth=1)
            else:
                ax.plot(x_coords, y_coords, color="pink", alpha=0.5, linewidth=1)



    # Add labels and legend
    ax.set_xlabel("Module Regions")
    ax.set_ylabel("Trophic Level")
    if name:
        ax.set_title(f"Trophic Web Segmented by Modules: {name}")
    else:
        ax.set_title("Trophic Web Segmented by Modules")

    # Add a legend for basal and apex species
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=basal_color, linestyle="None", markersize=10, label="Basal Species"),
        plt.Line2D([0], [0], marker="o", color=apex_color, linestyle="None", markersize=10, label="Apex Species"),
    ]
    for module_index in range(num_modules):
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color=colors[module_index % len(colors)], linestyle="None", markersize=10, label=f"Module {module_index}")
        )
    ax.legend(handles=legend_handles, title="Legend", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout
    plt.tight_layout()
    plt.show()

def artificial_apex(g, num_apex = 1): #Works by address, doesn't return a graph, but modifies the original graph
    
    #Adding two artificial apex predators to the antarctica graph to make in/out ratio 2:1, inverse of 1:2 arctic ratio
    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################
    # Identify the smallest module (grouped by leading eigenvector)
    leading_eigenvector_community = g.community_leading_eigenvector()
    leading_membership = leading_eigenvector_community.membership
    num_modules = max(leading_membership) + 1

    # Find the avg module
    module_sizes = [leading_membership.count(i) for i in range(num_modules)]

    # Calculate the average module size
    average_module_size = np.mean(module_sizes)

    # Find the module closest to the average size
    closest_module = min(
        range(num_modules),
        key=lambda i: abs(module_sizes[i] - average_module_size)
    )

    # Get the vertices in the smallest module
    average_module_vertices = [v.index for v in g.vs if leading_membership[v.index] == closest_module]

    # Calculate the average degree of apex predators
    apex_predators = [v.index for v in g.vs if g.degree(v.index, mode="in") > 0 and g.degree(v.index, mode="out") == 0]
    average_apex_degree = np.mean([g.degree(v, mode="in") for v in apex_predators])

    # Calculate the average in-module to out-module connection ratio for the smallest module
    in_out_ratios = []
    for v in average_module_vertices:
        in_module = sum(1 for neighbor in g.neighbors(v, mode="in") if leading_membership[neighbor] == closest_module)
        out_module = sum(1 for neighbor in g.neighbors(v, mode="out") if leading_membership[neighbor] != closest_module)
        if out_module > 0:
            in_out_ratios.append(in_module / out_module)
        else:
            in_out_ratios.append(0)  # Avoid division by zero

    average_in_out_ratio = np.mean(in_out_ratios)

    # Identify prey that are not apex predators and within the trophic level range
    trophic_levels = trophic_level_indexer(g)

    apex_trophic_levels = [trophic_levels[v] for v in apex_predators]

    # Calculate the average trophic level of apex predators
    average_apex_trophic_level = np.mean(apex_trophic_levels)

    prey_candidates_within_module_fixed = [
        v.index for v in g.vs
        if v.index not in apex_predators
        and v.index in average_module_vertices
        and -1.5 <= trophic_levels[v.index] - average_apex_trophic_level <= 0
    ]

    prey_candidates_outside_module_fixed = [
        v.index for v in g.vs
        if v.index not in apex_predators
        and v.index not in average_module_vertices
        and -1.5 <= trophic_levels[v.index] - average_apex_trophic_level <= 0
    ]

    for i in range(num_apex):

        # Add the artificial apex predator
        g.add_vertex(name=f"Artificial Apex Predator {i+1}")
        artificial_apex_index = g.vcount() - 1

        # Target sum
        target_sum = int(average_apex_degree)

        # Initialize variables to track the best pair
        best_x, best_y = None, None
        min_diff = float("inf")

        # Iterate over possible values of x and y
        for x in range(1, target_sum):  # x must be at least 1
            y = target_sum - x  # Ensure x + y = target_sum
            if y > 0:  # y must also be positive
                ratio = x / y
                diff = abs(ratio - average_in_out_ratio)  # Difference from the target ratio
                if diff < min_diff:  # Update the best pair if this is closer
                    best_x, best_y = x, y
                    min_diff = diff
        
        prey_candidates_within_module = prey_candidates_within_module_fixed.copy()
        prey_candidates_outside_module = prey_candidates_outside_module_fixed.copy()

        for i in range (int(best_x)):
            # Add edges from the artificial apex predator to prey candidates within the module
            random_index = random.randint(0, len(prey_candidates_within_module) - 1)

            g.add_edge(prey_candidates_within_module[random_index], artificial_apex_index)
            prey_candidates_within_module.remove(prey_candidates_within_module[random_index])

        for i in range (int(best_y)):
            # Add edges from the artificial apex predator to prey candidates outside the module
            random_index = random.randint(0, len(prey_candidates_outside_module) - 1)
            g.add_edge(prey_candidates_outside_module[random_index], artificial_apex_index)
            prey_candidates_outside_module.remove(prey_candidates_outside_module[random_index])

    #######################################################################
    #######################################################################
    #######################################################################
    #######################################################################

def write_dict_to_txt(dictionary, file_path):
    """
    Writes a dictionary to a text file.

    Args:
        dictionary (dict): The dictionary to write.
        file_path (str): The path to the text file.
    """
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def read_dict_from_txt(file_path):
    """
    Reads a dictionary from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        dict: The dictionary read from the file.
    """
    dictionary = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(": ", 1)  # Split on ": " to separate key and value
            dictionary[key] = value  # Use eval to handle non-string values (e.g., lists, numbers)
    return dictionary


def load_food_webs():
    """Load Antarctic (Potter Cove) and Arctic (Kongsfjorden) food-web graphs."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(repo_dir, "Potter_FW.txt")
    csv_converted_path = os.path.join(repo_dir, "Potter_FW.csv")

    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(csv_converted_path, "w") as csv_file:
        for line in lines:
            csv_file.write(line.replace("\t", ","))

    predation = pd.read_csv(csv_converted_path)
    predation = predation.drop_duplicates()

    names = set()
    for row in predation.itertuples():
        names.update(row[1:])
    names = list(names)

    edges = []
    for row in predation.itertuples():
        edges.append([names.index(row[2]), names.index(row[1])])

    antarctica = ig.Graph(edges=edges, directed=True)
    antarctica.vs["name"] = names

    file_path = os.path.join(repo_dir, "Arctic_Food_Web_2016.xlsx")
    binary_predation_matrix = pd.read_excel(file_path)
    csv_converted_path = os.path.join(repo_dir, "Arctic_Food_Web_2016.csv")

    with open(csv_converted_path, "w") as csv_file:
        csv_file.write("Predator,Prey\n")
        for j in range(1, len(binary_predation_matrix.columns)):
            for i in range(0, len(binary_predation_matrix.columns) - 1):
                if binary_predation_matrix.iloc[i, j] == 1:
                    csv_file.write(
                        f"{binary_predation_matrix.columns[j]},"
                        f"{binary_predation_matrix.iloc[i, 0]}\n"
                    )

    predation = pd.read_csv(csv_converted_path)
    names = set()
    for row in predation.itertuples():
        names.update(row[1:])
    names = list(names)

    edges = []
    for row in predation.itertuples():
        edges.append([names.index(row[2]), names.index(row[1])])

    arctic = ig.Graph(edges=edges, directed=True)
    arctic.vs["name"] = names

    return antarctica, arctic


def run_extinction_trace(g, metric, type_, threshold=0.5):
    """Run one dynamic extinction simulation and return trajectory data."""
    initial_count = g.vcount()
    temp_g = g.copy()
    x_steps = list(range(initial_count))
    y_counts = []
    by_species = {name: [] for name in g.vs["name"]}
    collapse = 0
    max_delta = 0
    mark = None
    popped = None
    pop_list = []

    for i in x_steps:
        trophic_level_indexer(temp_g)
        ksi_indexer(temp_g)
        for j in range(temp_g.vcount()):
            by_species[temp_g.vs[j]["name"]].append(temp_g.vs[j]["ksi"])

        components = temp_g.components(mode="weak")
        if (
            len(components) > 1
            and collapse == 0
            and all(x > 1 for x in sorted(components.sizes(), reverse=True)[0:2])
        ):
            collapse = i

        y_counts.append(temp_g.vcount())

        if len(y_counts) > 1:
            pop_list.append(popped)
            if y_counts[-2] - y_counts[-1] > max_delta:
                max_delta = y_counts[-2] - y_counts[-1]
                mark = (popped, i - 1)

        temp_g, popped = extinction_simulation(
            temp_g, type_=type_, metric=metric, percent=0, threshold=threshold
        )

    pop_list.append("END")
    x_pct = [x / initial_count * 100 for x in x_steps]
    y_pct = [y / initial_count * 100 for y in y_counts]

    return {
        "x_pct": x_pct,
        "y_pct": y_pct,
        "counts": y_counts,
        "by_species_ksi": by_species,
        "pop_list": pop_list,
        "initial_count": initial_count,
        "collapse_index": collapse,
        "mark": mark,
        "r50": robustness_50(y_counts, initial_count),
    }


def robustness_50(population_history, initial_count):
    """Calculate Robustness-50 index for an extinction simulation.

    Parameters
    ----------
    population_history : list of int
        Remaining species counts after each primary extinction step. The first
        value should correspond to the initial number of species.
    initial_count : int
        Number of species at the beginning of the simulation.

    Returns
    -------
    float
        The Robustness-50 index defined as the proportion of primary
        extinctions required to drive the community to 50% of its original
        size. A value of 0.5 indicates no secondary extinctions.
    """

    half = initial_count * 0.5
    for i, remaining in enumerate(population_history):
        if remaining <= half:
            return i / initial_count
    # Population never dropped below half
    return len(population_history) / initial_count
