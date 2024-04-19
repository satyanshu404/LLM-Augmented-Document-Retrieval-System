import os
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set CUDA devices and PyTorch environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"]="1" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

api_key = "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"
file_path = "Dataset/triples.tsv.gz"
final_file_path = 'Dataset/triples_with_justifications.tsv.gz'
model_checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
top_k = 500  # no. of datapoints (each datapoint will generate justifications for rel and nrel doc)
max_new_tokens = 1000

# login to hugging face
HfFolder.save_token(api_key)

# Load the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, device_map=device)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map=device)

# Load the dataset
df = pd.read_csv(file_path,sep='\t', encoding='utf8')

# add two new colums for justifications
df['nrdocJusti'] = [None]*len(df.index)
df['rdocJusti'] = [None]*len(df.index)

# prompt function
def message(query:str, doc:str, relevancy:str):
    
    if relevancy == "Non-Relevant":
        user_content = '''
            Query:  _________ justice is designed to repair the harm to victim, the community and the offender caused by the offender criminal act.
            Document: Sex Offender Registration: Policy Overview and Comprehensive Practices October 1999Introduction Highly publicized sex crimes committed by repeat offenders in recent years have prompted state legislatures to ratify laws that increase social controls on these offenders. State and federal laws have been enacted that require released sex offenders to register with law enforcement or other state agencies. Registration laws require offenders to supply their addresses and other identifying information to a state agency or law enforcement. These laws have been enacted in every state. In the United States, offender registration was first used in the 1930s. These early ordinances focused on repeat criminal offenders as well as sex offenders. They operated as a means to drive out persons who were undesirable (Lieb et al, 1998). Current registration statutes make sex offenders more visible to law enforcement and the public, with the intent of increasing public safety. The first of these current sex offender registration statutes was enacted in California in 1947. This document compares state registration statutes and identifies promising and comprehensive practices in use throughout the country. Background In 1994, Congress passed the Jacob Wetterling Crimes Against Children and Sexually Violent Offender Registration Act (Title XVII of the Violent Crime Control and Law Enforcement Act of 1994 [42 U. S. C. A. § 14071]). The Act requires states to create registries of offenders convicted of sexually violent offenses or crimes against children and to establish heightened registration requirements for highly dangerous sex offenders. It further requires offenders to verify their addresses annually for a period of 10 years and requires sexually violent predators to verify addresses on a quarterly basis for life. States that do not establish registration programs, in compliance with the Act’s provisions, are subject to a 10 percent reduction of Byrne formula grant funding (The Edward Byrne Memorial State and Local Law Enforcement Assistance Programs [42 U. S. C. § 3750]). Any such funds will be reallocated to states that are in compliance (the Office of Justice Programs, U. S. Department of Justice, is monitoring states’ efforts to comply with the Wetterling Act). The vast majority of states have enacted sex offender registration laws within the last 15 years. Since 1991, 38 of the 50 states (and the District of Columbia) have passed laws. Many amendments have passed since 1994 to bring state legislation into compliance with federal law. This document does not describe states’ efforts to comply with the Wetterling Act; rather, it describes states’ current registration practices. State legislation concerning sex offenders has been highly active in recent years, with some state legislatures amending their laws annually in an attempt to meet requirements of the Wetterling Act. Therefore, information contained in this document was current at the time of printing. Goals of Registration Sex offender registration statutes are promoted as a means of:deterring offenders from committing future crimes;providing law enforcement with an additional investigative tool; andincreasing public protection. Alabama’s registration statute describes the reasoning behind these goals: "The Legislature further finds that law enforcement agencies’ efforts to protect their communities, conduct investigations, and quickly apprehend criminal sex offenders are impaired by the lack of information about criminal sex offenders who live within their jurisdiction and that the lack of information shared with the public may result in the failure of the criminal justice system to identify, investigate, apprehend, and prosecute criminal sex offenders." Alabama Enacted Laws, 1998 General Session: Act 98-489. Deterring Offenders from Committing Future Crimes. Sex offender registries identify released sex offenders living in a specific area and summarize their offense histories. Law enforcement officials can monitor these offenders, requiring them to update their addresses if they move, or their names if they are changed. Registries also contain information on patterns of offending behavior, which can assist criminal justice agents with identifying "risky" situations (e.g., a child molester working in a daycare center or residing near children). Police officers in Illinois utilize the registry during routine traffic stops. In addition to conducting a criminal history check on drivers, officers will find out whether drivers are registered sex offenders and if they must comply with any specific release conditions (e.g., staying at least 1,000 feet from schools or daycare centers). Providing Law Enforcement with an Additional Investigative Tool. Registration can be used to assist in investigations. The detailed information that law enforcement receives on sex offenders can potentially identify likely suspects with similar crime patterns for unsolved sex offenses. Increasing Public Protection. Citizens are afforded assistance in protecting themselves from convicted sex offenders through public access to registries and active dissemination of registration information by criminal justice officials. Community notification policies grew out of this goal (Center for Sex Offender Management, 1997). All 50 states, and the District of Columbia, have enacted some type of community notification, commonly referred to as Megan’s Law (at the time of this printing, Pennsylvania’s Megan’s Law was enjoined due to a June 30, 1999 Pennsylvania Supreme Court decision [ Commonwealth of Pennsylvania v. D. F. Williams ]). Federal Requirements The original compliance deadline for the Jacob Wetterling Act was September 1997; a two-year extension was granted to states making good faith efforts to achieve compliance. States granted this extension had until September 12, 1999, to comply with the original features of the Act. States were not allocated any additional federal funding to achieve these efforts—and risked losing crime control funds if found not in compliance. The Wetterling Act has been amended three times: Megan’s Law, 1996 (requiring community notification); the Pam Lychner Sexual Offender Tracking and Identification Act of 1996 (heightening registration requirements for repeat and aggravated offenders); and section 115 of the General Provisions of Title I of the Departments of Commerce, Justice, and State, the Judiciary, and Related Agencies Appropriations Act, 1998 (CJSA) (amending sexually violent predator provisions and adding registration of federal and military sex offenders, and sex offenders who are non-resident students or workers). Guidelines have been issued to assist states with complying with the Act and its amendments (U. S. Department of Justice, 1999). Wetterling Act The original requirements of the Wetterling Act created several conditions, including: registering offenders for at least 10 years; acquiring registration information from offenders when they are released and informing them of registration obligations in jurisdictions where they intend to reside; requiring registrants to update address information when they move; verifying registered addresses periodically; and releasing registration information as necessary for public safety. Megan’s Law Megan’s Law amended the Wetterling Act in May 1996 by requiring that "the state or any agency authorized by the state shall release relevant information as necessary to protect the public" concerning a specific sex offender. Megan’s Law allows states discretion in determining if disclosure of information is necessary for public protection. It also allows states discretion in specifying standards and procedures for making these determinations. 
            Relevancy: Non-Relevant
            Output:
        '''
        assistant_content = '''
            The document focuses on sex offender registration policies, detailing their history, purpose, and implementation across the United States. It does not discuss or mention the concept of justice aimed at repairing harm to victims, communities, and offenders caused by criminal acts, which is the central focus of the query.
        '''
    else:
        user_content = '''
            Query: what was the immediate impact of the success of the manhattan project?
            Document: The Manhattan Project Introduction Importance of Core Engineering Disciplines to the Manhattan Project Significance of Communication Morality of the Manhattan Project Conclusion References Abstract The pivotal engineering and scientific success of the Twentieth century was the Manhattan Project. The Manhattan Project assimilated concepts and leaders from all scientific fields and engineering disciplines to construct the first two atomic bombs. From the study of nuclear physics and chemistry to the practical engineering and processing of uranium 235 and plutonium 239 and the final construction of the weapons, scientific knowledge grew at an exponential rate to critical levels. The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated. Yet this grave definition of success cannot diminish the impressive collaboration and efficiency of the Manhattan Project. Index Terms   Atomic Bomb, Fat Man, Little Boy, Manhattan Project, Oppenheimer, J. Robert. Introduction The Manhattan Project was the American program for researching and developing the first atomic bombs. The weapons produced were based solely upon the principles of nuclear fission of uranium 235 and plutonium 239, chain reactions liberating immense amounts of destructive heat energy. Although originally established in Manhattan, New York by the Manhattan Engineer District of the U. S. Army Corps of Engineers, the majority of the research took place under director General Leslie Groves at the Los Alamos laboratory in New Mexico. The goal of the Manhattan Project was effectively summed up by scientist Robert Serber when he deduced,  Since the one factor that determines the damage is the energy release, our aim is simply to get as much energy from the explosion as we can.  [1] Thus, due to the nature of the program s objective, the Manhattan Project is one of scientific engineering s foremost successes. In the quest for an atomic-powered weapon, the secrets of nuclear physics and chemistry were exposed. Following the theoretical assessment of producing a controllable nuclear chain reactor, physical engineering was employed to construct the specific mechanics required. Communication contributed as much to the success of the Manhattan Project as did scientific discovery. Although the creation of the first atomic weapon was clearly a technological triumph, the question of morality and responsibility to ethics will forever plague the topic. Regardless of whether America was morally justified in deploying atomic weaponry on Japan, though, the Manhattan Project will always be an excellent example of collaboration and communication in scientific and engineering fields. Importance of Core Engineering Disciplines to the Manhattan Project A little bomb like that,  declared physicist Enrico Fermi, enthralled by his first taste of nuclear fission,  and it would all disappear.  [2]. The Atomic Age, a period of incessant discovery and revelation of atomic and subatomic wonders - an age that revolutionized the physical world - began on a vacant playing field beneath the University of Chicago stadium on December 2, 1942. In the late afternoon of this momentous day, Fermi and Leo Szilard created the first controlled nuclear reactor, a model later reconstructed into five different reactor prototypes. [3] From the first controllable chain reaction to the dropping of atomic weapons on Hiroshima and Nagasaki, Japan, the fields of physics, chemistry, and mathematics - the core disciplines of modern engineering   raced mercilessly ahead to godly enlightenment: the power of life and annihilation. The first atomic bomb, a weapon harnessing the devastating power of nuclear fission, was developed as an end to World War II and all war thereafter. Comprehension of the bomb and its historical development is attained by breaking the subject into three related components: chemistry, nuclear physics, and the practical engineering that realized the theoretical dream. The Chemistry Aspect Fission is an elementary chemical interaction between subatomic particles. Nuclear fission is defined as the splitting of an atom by nucleus bombardment. Atoms consist of three subatomic particles: negatively charged electrons, positively charged protons, and neutrons, which have no electrical charge. Atomic nuclei are dense cores of atoms composed of neutrons and protons, and are thus positively charged. Chemical reactions, from basic acid-base titrations to nuclear fission, involve the collision of atomic particles. Fission begins with the high-energy collision of neutrons with the nucleus of another atom. Protons cannot partake in nuclear bombardment because of the electrostatic repulsion between positively-charged protons and nuclei. For fission to proceed, a neutron fired at the atom must fuse with the nucleus, producing a less-stable isotope. The  heavy  atom, chemically volatile, will split into two stable atoms, discharge neutrons, and generate energy (in the form of Gamma radiation). The neutrons released are free to collide and fuse with nuclei of other nearby atoms   a chain reaction ensues, progressing exponentially throughout the sample of atoms, releasing more and more heat radiation. It is this constant amplification of energy that constitutes the devastating power of an atomic weapon. If every atom were fissionable, there would be no stability to matter and the world would be an uninhabitable chaos of energy transformations. Thus, only certain isotopes of few atoms will undergo a fissile chain reaction. Nobel physicist Neils Bohr discovered that Uranium, the ninety-second element, is an example of a fissionable atom. [4] the predominant form of uranium has a mass number of 238 and is relatively stable; U-235, a rare isotope, easily undergoes nuclear fission and can sustain a chain reaction. If a neutron traveling at adequate velocity and the appropriate angle collides with a uranium-235 nucleus, the unstable nucleus absorbs the neutron, consequently increasing to an exceedingly unstable state. Instantly the U-236 splits into two atoms of different elements, emitting 2 neutrons that carry on the chain reaction, and liberating large quantities of gamma radiation. Subsequent to John Dunning and Eugene Booth s successful separation of uranium 235 from uranium 238 in 1941 [5], uranium became the leading example for fission research. As scientists probed further into the capabilities of fission, they began to visualize, as Fermi did, the awesome destructive potential of only a minute mass of uranium 235. Thus, the dream of a weapon with unmitigated atomic power was spawned. With this basic knowledge of atomic chemistry and the motivation of a world crisis, the Manhattan project began in 1942. Two prominent challenges delayed the successful unleashing of nuclear energy: processing of sufficient fissionable material, and the actual design of a nuclear bomb that would maintain and maximize a fissile chain reaction. Dunning and Booth managed to separate the two isotopes of uranium before the institution of the Manhattan project, proving that it was indeed possible. But the level at which their research was performed was purely scientific; a project tasked with ending a world war required much more intensive and efficient processing. Additionally, no one had any experience constructing a dimensionally-feasible nuclear reactor in deployable size. As the Manhattan team quickly learned, the relative abundance of uranium 235 was not only extremely small, but when found, uranium ore was nearly impossible to purify into U-235. Natural uranium ore is comprised of a mixture of isotopes 235 and 238. Typically, one percent of the mass of uranium ore being considered is composed of the unstable isotope, while the rest is relatively useless U-238. [6] Therefore, to develop the prophesized weapon, engineering innovations were essential. Ernest O. Lawrence of the University of California, Berkeley developed the first technique to isolate a practical amount of uranium 235, using a modified mass spectrometer. [7] Lawrence s method dealt with the electromagnetic properties of atoms, an integration of chemistry and physics. Uranium 235 is a lighter isotope than Uranium 238. Thus, when equal physical forces act on atoms of U-235 and U-238, the isotopes will behave simply as two different masses. Therefore, as Newton s laws describe, the force acting on the lighter mass will show greater affect on the motion of the mass than the force acting on the heavier mass. In Lawrence s apparatus, the magnetic force exerted by an electromagnetic arch attracted the electrically charged portions of atoms, and thus, uranium atoms traveled along the raised arc. Uranium 235 atoms, naturally lighter than the stable isotope, would pass closer to the magnetic tract than the heavier U-238 atoms, and could thus be collected. To mobilize the uranium atoms into gaseous particles, fluorine gas was circulated over solid uranium ore, creating uranium tetrachloride gas. The gaseous particles were then passed across the arc multiple times, each trial gaining a more purified sample of uranium 235 tetrachloride. To Lawrence s chagrin, his method of uranium purification proved largely inefficient. Due to high maintenance costs and the time-consuming nature of the process, only one gram of uranium 235 was harvested after millions of dollars spent in production and repairs. [8] Lawrence s design was quickly abandoned. General Leslie Groves, director of the Manhattan Project, purchased land in Oak Ridge, Tennessee to pursue another proposed method of uranium separation. Centered on the theory of gaseous diffusion of uranium hexafluoride, principles of chemistry and physics were again assimilated in an attempt to accomplish a feat of engineering. Similarly, the technique Groves pursued dealt with the motions of particles of different masses. When contained in a porous vessel, gaseous particles will randomly collide and bounce until passing through a pore in the container   a process known as effusion. At constant temperature, all gaseous particles have equal kinetic energies, and therefore, through the properties of thermodynamics, less-massive particles move with higher velocity than heavier particles. When applied to effusion, lighter gases travel across a porous membrane faster than heavier gases. Uranium 235 hexafluoride, the lighter of the two isotopic gases, diffuses faster than U-238 hexafluoride. Over intervals of time, Groves concluded, this process could be used to separate uranium isotopes. The foremost problem with this technique was the quality of materials used in construction. The masses of the two isotopes of uranium are so similar that the diffusion must be run in completely airtight conditions. Grease could not be used to seal the miles of tubing, however, because uranium hexafluoride reacts so quickly with organic compounds. Thus, materials engineering was incorporated to fabricate new plastics to assemble the contraption. The engineering efforts resulted in an effective material known as Teflon. [9] Higher quality porous membranes were essential to the process as well. The final product was a $100,000,000 facility consisting of thousands of consecutive membranes and diffusion tanks across miles of Teflon tubing   an engineering masterpiece. [10] Due to restrictions of the budget, however, construction on the Oak Ridge plant was only carried far enough to produce a mixture of the two isotopes, half stable, half unstable. [11] This end product was subsequently fed into other series of separation techniques, becoming the main source of U-235. Following Bohr s discovery of fissionable uranium in 1941, Glen Seaborg capitalized on years of research by the discovery of the ninety-fourth element, plutonium.
            Relevancy: Relevant
        '''
        assistant_content = '''
            The document provides a comprehensive overview of the Manhattan Project. This directly addresses the query about the immediate impact of the project's success, highlighting the exponential growth in scientific knowledge and the moral questions raised by the project's outcome. The mention of the project's scientific and engineering triumphs, along with its ethical implications, captures the immediate impact on both technological advancement and societal perspectives on atomic energy.
        '''
    
    messages = [
            {"role": "user", "content": '''Determine whether the given document is relevant or not to the given query. Justify your decision by considering below cases.
                - If the document is relevant to the query, identify and mention briefly the specific keywords or phrases in the document that support its relevance to the query.
                - If the document is not relevant to the query, explain briefly why and highlight the aspects or keywords in the document that contradict or fail to support the query.
            '''},
            {"role": "assistant", "content": '''Sure, I'll follow these instructions.
            '''},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": f'''
                Query:  {query}
                Document: {doc}
                Relevancy: {relevancy}
                Output:
            '''}
        ]
    return messages

# function to generate justifications for one relevant and one non-relevant
def generate_justification(query: str, doc:str, relevancy:str) -> str:
    
    messages = message(query, doc, relevancy)
    prompt = f"Query: {query}\nDoc:{doc}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print('Query+Doc token len: ', input_ids.shape)
    if input_ids.shape[-1] > 15_500:
        return None #type:ignore
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print('Query+Doc+Examples token len',encodeds.shape) #type:ignore
    model_inputs = encodeds.to(device) #type:ignore
    model.to(device)

    with torch.no_grad():
        generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
    res = tokenizer.batch_decode(generated_ids)[0]

    return res


for idx in tqdm(range(top_k)):
    # for relevant document
    query = df.iloc[idx]['query']
    doc = df.iloc[idx]['rdoc']
    relevancy = 'Relevant'

    res = generate_justification(query, doc, relevancy)
    if res is not None:
        res = res.split('[/INST]')[-1]
        df.at[idx, 'rdocJusti'] = res

    # for non-relevant document
    query = df.iloc[idx]['query']
    doc = df.iloc[idx]['ndoc']
    relevancy = 'Non-Relevant'

    res = generate_justification(query, doc, relevancy)
    if res is not None:
        res = res.split('[/INST]')[-1]
        df.at[idx, 'nrdocJusti'] = res

    df.to_csv(final_file_path, sep='\t', compression='gzip', index=False)


# Save the DataFrame to a TSV file compressed with gzip
df.to_csv(final_file_path, sep='\t', compression='gzip', index=False)


# Testing
print("Test Example")
for i in range(0, 1):
    print(df.iloc[i]['query'])
    print(df.iloc[i]['rdocJusti'])
    print(df.iloc[i]['nrdocJusti'], '\n')

print("-- All the justification generated sucessfully --")
