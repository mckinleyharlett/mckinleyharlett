# McKinley Harlett
# Create a Python Application which asks the user for their zip code or city.
# Use the zip code or city name in order to obtain weather forecast data from OpenWeatherMap.
# Display the weather forecast in a readable format to the user.
# Use comments within the application where appropriate in order to document what the program is doing.
# Use functions including a main function.
# Allow the user to run the program multiple times to allow them to look up weather conditions for multiple locations.
# Validate whether the user entered valid data. If valid data isn’t presented notify the user.
# Use the Requests library in order to request data from the webservice.
# Use Try blocks to ensure that your request was successful. If the connection was not successful display a message to the user.
# Use try blocks when establishing connections to the webservice.
# You must print a message to the user indicating whether or not the connection was successful

import requests
import simplejson as json
import zipcodes
import math


def weather_for_zip():

    # Print that the URL worked
    print("Successful Zipcode!")
    z_data = x.text
    z_parsed = json.loads(z_data)

    # Pull out the description, temp, temp_min, temp_max, humidity, zip_code, name
    z_temp = z_parsed['temp']
    z_temp_max = z_parsed['temp_max']
    z_temp_min = z_parsed['temp_min']
    z_humidity = z_parsed['humidity']
    z_city_zip_name = z_parsed['name']
    z_description = z_parsed['description']

    # Converting kelvin to fahrenheit
    z_kelvin_temp = z_temp
    # temperature in celsius stored in variable `celsius_temp`
    z_celcisus_temp = z_kelvin_temp - 273.15
    # convert celsius to fahrenheit stored in variable `fahrenheit_temp`
    z_fahrenheit_temp = z_celcisus_temp * (9 / 5) + 32
    # round the value of fahrenheit down and assign to `fahrenheit_temp`
    z_fahrenheit_temp = math.floor(z_fahrenheit_temp)

    # Print the weather report
    print("Here is your weather report for " + zip_code)
    print("City: " + z_city_zip_name)
    print("ZipCode: " + zip_code)
    print("Temperature: " + z_fahrenheit_temp)
    print("High Temp of Today: " + z_temp_max)
    print("Low Temp of Today: " + z_temp_min)
    print("Humidity: " + z_humidity)
    print("Description: " + z_description)

def weather_for_city():

    # Print that the URL worked
    print("Successful City Name!")

    c_data = y.text
    parsed = json.loads(c_data)

    # Pull out the description, temp, temp_min, temp_max, humidity, zip_code, name
    c_temp = parsed["temp"]
    c_temp_max = parsed["temp_max"]
    c_temp_min = parsed["temp_min"]
    c_humidity = parsed["humidity"]
    c_decsription = parsed["description"]

    # Converting kelvin to fahrenheit
    c_kelvin_temp = c_temp
    # temperature in celsius stored in variable `celsius_temp`
    c_celcisus_temp = c_kelvin_temp - 273.15
    # convert celsius to fahrenheit stored in variable `fahrenheit_temp`
    c_fahrenheit_temp = c_celcisus_temp * (9 / 5) + 32
    # round the value of fahrenheit down and assign to `fahrenheit_temp`
    c_fahrenheit_temp = math.floor(c_fahrenheit_temp)

    # Print the weather report
    print("Here is your weather report for " + cityname)
    print("City: " + cityname)
    print("Temperature: " + c_fahrenheit_temp)
    print("High Temp of Today: " + c_temp_max)
    print("Low Temp of Today: " + c_temp_min)
    print("Humidity: " + c_humidity)
    print("Description: " + c_decsription)



if __name__=='__main__':

    # Want to get user to say c or z
    while True:
        goagain = input("Would you like to know the weather for a city or zipcode? Y for yes or N for no: ")
        goagain = goagain.lower()

        # If yes
        if goagain == 'y':
            zip_or_cityname = input("Are you going to enter a city name or zipcode? (C for city, Z for Zip)*Note: more accurate would be by Zip: ")
            zip_or_cityname = zip_or_cityname.lower()
            # If they select a Zipcode
            if zip_or_cityname == 'z':
                # Gets the zipcode
                zip_code = input("Enter your desired Zipcode: ")
                # Enter your API key here
                api_key = "64d58f18e6695b60690bf408a4b5de65"
                # Base URL
                base_url = "http://api.openweathermap.org/data/2.5/weather?"
                # Completed URL so I can get the zipcode
                z_complete_url = (base_url + "zip=" + zip_code + "&appid=" + api_key)
                # Making sure the Zipcode is correct
                try:
                    x = requests.get(z_complete_url)
                except requests.ConnectionError:
                    print("Your Zipcode is not valid!")

                # if it passes it will go to the zipcode
                weather_for_zip()

            elif zip_or_cityname == 'c':
                # Get the City Name
                cityname = input("Enter your desired City Name: ")
                # Enter your API key here
                api_key = "64d58f18e6695b60690bf408a4b5de65"
                # Base URL
                base_url = "http://api.openweathermap.org/data/2.5/weather?"
                # Completed URL so I can get the city
                c_complete_url = (base_url + "q=" + cityname + "&appid" + api_key)
                # Make sure the URL works
                try:
                    y = requests.get(c_complete_url)
                except requests.ConnectionError:
                    print("Your City Name is not valid!")
                weather_for_city()
            else:
                print("Please enter C or Z: ")
        elif goagain == 'n':
            print("Thank you and have a nice day!")
            break
        else:
            print("Please enter Y or N")



