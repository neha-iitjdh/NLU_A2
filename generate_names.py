import random

male_names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Shaurya", "Atharva", "Advik", "Pranav", "Advaith",
    "Aaryan", "Dhruv", "Kabir", "Ritvik", "Anirudh", "Rohan", "Arnav", "Sahil",
    "Karthik", "Nikhil", "Rahul", "Amit", "Vikram", "Suresh", "Rajesh",
    "Mohan", "Ganesh", "Siddharth", "Manish", "Pradeep", "Ravi", "Sunil",
    "Deepak", "Vinay", "Ajay", "Vijay", "Arun", "Varun", "Tarun", "Naveen",
    "Pavan", "Kiran", "Manoj", "Anand", "Ramesh", "Dinesh", "Mukesh",
    "Rakesh", "Lokesh", "Yogesh", "Hitesh", "Jitesh", "Nilesh", "Paresh",
    "Naresh", "Mahesh", "Umesh", "Rupesh", "Alpesh", "Brijesh", "Devesh",
    "Ritesh", "Kamlesh", "Sudhir", "Satish", "Girish", "Harish", "Jagdish",
    "Ashish", "Mandeep", "Sandeep", "Prateek", "Abhishek", "Animesh", "Avnish",
    "Tushar", "Gaurav", "Sourav", "Anurag", "Chirag", "Neeraj", "Pankaj",
    "Vikas", "Ankur", "Mayank", "Hemant", "Sumit", "Mohit", "Rohit", "Lalit",
    "Nitin", "Sachin", "Vaibhav", "Saurabh", "Rishabh", "Harsh", "Darsh",
    "Yash", "Akash", "Prakash", "Subhash", "Avinash", "Devendra", "Narendra",
    "Surendra", "Mahendra", "Rajendra", "Jitendra", "Virendra", "Upendra",
    "Dharmendra", "Gajendra", "Nagendra", "Ravindra", "Satendra", "Bhupendra",
    "Akhil", "Nihal", "Vishal", "Kushal", "Vimal", "Kamal", "Nishant",
    "Prashant", "Hemanshu", "Himanshu", "Siddhartha", "Parth", "Parthiv",
    "Dev", "Deven", "Manan", "Darshan", "Lakshay", "Tanmay", "Chinmay",
    "Tejas", "Ojas", "Shivam", "Gautam", "Uttam", "Satyam",
    "Shubham", "Soham", "Param", "Pratham", "Ishan", "Kishan",
    "Bhushan", "Roshan", "Mihir", "Samir", "Amir",
    "Kabeer", "Zaheer", "Tanveer", "Ranveer", "Jasveer",
    "Rajveer", "Manveer", "Gunveer",
    "Abhinav", "Raghav", "Madhav", "Keshav", "Vasudev", "Uday",
    "Ajit", "Sujit", "Ranjit", "Manjit", "Daljit", "Surjit", "Harjit",
    "Paramjit", "Jagjit", "Kuldeep", "Sukhdeep", "Hardeep",
    "Bharat", "Aryan", "Shreyas", "Aneesh",
    "Ashwin", "Praveen", "Chetan", "Ketan",
    "Rajan", "Pawan", "Jeevan", "Kalyan", "Charan",
    "Sagar", "Shankar", "Omkar", "Onkar",
    "Bhaskar", "Divakar", "Pushkar",
    "Shekhar", "Sohan", "Lakhan", "Krishan",
    "Sudhanshu",
]

female_names = [
    "Aadhya", "Saanvi", "Aanya", "Aaradhya", "Ananya", "Pari", "Anika",
    "Navya", "Diya", "Myra", "Sara", "Iraa", "Ahana", "Anvi",
    "Prisha", "Riya", "Aarohi", "Anaya", "Akshara", "Shanaya",
    "Kiara", "Siya", "Tara", "Divya", "Kavya", "Shreya", "Meera", "Nisha",
    "Pooja", "Neha", "Sneha", "Priya", "Pallavi", "Anjali", "Sonali",
    "Deepali", "Rupali", "Shweta", "Swati", "Jyoti", "Preeti", "Smriti",
    "Kriti", "Aditi", "Sruti", "Garima", "Pratima", "Karishma", "Reshma",
    "Padma", "Durga", "Radha", "Sudha", "Vidya", "Ramya", "Tanvi", "Manvi",
    "Janvi", "Devi", "Lakshmi", "Saraswati", "Parvati", "Gauri",
    "Bhavani", "Vaishnavi", "Madhavi", "Radhika", "Chandrika", "Mallika",
    "Aishwarya", "Akanksha", "Aparna", "Archana", "Bhavna", "Chitra",
    "Deepika", "Ekta", "Geeta", "Hema", "Indira", "Jaya",
    "Kamala", "Lata", "Madhuri", "Nalini", "Padmini", "Revathi", "Savitri",
    "Usha", "Vanita", "Yamini", "Zara", "Asha", "Bharati", "Chhaya",
    "Damini", "Esha", "Falguni", "Ganga", "Harini", "Ila", "Jhanvi",
    "Kalpana", "Leela", "Mala", "Nandini", "Oviya", "Purnima", "Ragini",
    "Sarla", "Tanya", "Uma", "Varsha", "Yashoda", "Zoya",
    "Amrita", "Bindu", "Chandni", "Devika", "Ela", "Gouri",
    "Hiral", "Isha", "Komal", "Lavanya", "Mitali", "Namrata",
    "Prerna", "Rachna", "Shalini", "Tanuja", "Unnati", "Vibha",
    "Ankita", "Nikita", "Sanjana", "Rani", "Simran", "Mansi", "Sakshi",
    "Rashi", "Khushi", "Srishti", "Drishti", "Ishita", "Nidhi", "Riddhi",
    "Siddhi", "Samridhi", "Surabhi", "Urvashi", "Vartika",
    "Anamika", "Chandana", "Deeksha", "Harshita", "Kashish", "Meenal",
    "Payal", "Richa", "Shilpa", "Vasundhara", "Yamuna",
    "Arti", "Bharti", "Dipti", "Gayatri", "Juhi", "Kajal", "Laxmi",
    "Mamta", "Neelam", "Poonam", "Reema", "Seema", "Trupti", "Urmi",
]

last_names = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Joshi", "Patel",
    "Shah", "Mehta", "Desai", "Reddy", "Rao", "Nair", "Menon", "Pillai",
    "Iyer", "Iyengar", "Mukherjee", "Banerjee", "Chatterjee", "Ghosh",
    "Bose", "Sen", "Das", "Dutta", "Roy", "Saha", "Pal", "Mondal",
    "Mishra", "Pandey", "Tiwari", "Dubey", "Shukla", "Bajpai", "Srivastava",
    "Trivedi", "Dwivedi", "Chaturvedi", "Upadhyay", "Pathak", "Saxena",
    "Nigam", "Kapoor", "Chopra", "Malhotra", "Khanna", "Bhatia", "Arora",
    "Sethi", "Tandon", "Mehra", "Sehgal", "Dhawan", "Grover", "Anand",
    "Bedi", "Chawla", "Garg", "Goyal", "Jain", "Mittal", "Bansal",
    "Agarwal", "Maheshwari", "Khandelwal", "Somani", "Rathi", "Saini",
    "Yadav", "Chauhan", "Rajput", "Thakur", "Rathore", "Shekhawat",
    "Bhati", "Solanki", "Parmar", "Jadeja", "Gohil", "Chaudhary",
    "Malik", "Sangwan", "Dahiya", "Hooda", "Tanwar",
    "Nayak", "Patil", "Kulkarni", "Deshpande", "Gokhale",
    "Bhatt", "Vyas", "Raval", "Panchal", "Kothari",
    "Hegde", "Shetty", "Kamath", "Bhat", "Gowda", "Naidu",
    "Choudhury", "Barman", "Kalita", "Hazarika", "Gogoi", "Borah",
    "Swamy", "Murthy", "Prasad", "Subramaniam", "Krishnamurthy",
]

random.seed(42)
names = set()

while len(names) < 1000:
    if random.random() < 0.5:
        first = random.choice(male_names)
    else:
        first = random.choice(female_names)
    last = random.choice(last_names)
    full_name = first + " " + last
    names.add(full_name)

names = sorted(names)

with open("e:/NLU2/TrainingNames.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(name + "\n")

print(f"Generated {len(names)} unique Indian names")
