from functions import *

# create chatbot 
home_bot = create_bot('Jordan')

# train all data
train_all_data(home_bot)

# check identity
identity = input("State your identity please: ")

# rules for responding to different identities
if identity == "Mark":
    print("Welcome, Mark. Happy to have you at home.")

elif identity == "Jane":
    print("Mark is out right now, but you are welcome to the house.")

else:
    print("Your access is denied here.")
    exit()

# custom data
house_owner = [
    "Who is the owner of this house?",
    "Mark Nicholas is the owner of this house."
]
custom_train(home_bot, house_owner)

print("------ Training custom data ------")
# write and train your custom data here IF the identity is Mark
if identity == 'Mark':   
    city_born = [
        "Where was I born?",
        "Mark, you were born in Seattle."
    ]

    fav_book = [
        "What is my favourite book?",
        "That is easy. Your favourite book is The Great Gatsby."
    ]

    fav_movie = [
        "What is my favourite movie?",
        "You have watched Interstellar more times than I can count."
    ]

    fav_sports = [
        "What is my favourite sport?",
        "You have always loved baseball."
    ]
    # train chatbot with your custom data
    custom_train(home_bot, city_born)
    custom_train(home_bot, fav_book)
    custom_train(home_bot, fav_movie)
    custom_train(home_bot, fav_sports)

# start chatbot
start_chatbot(home_bot)