from bs4 import BeautifulSoup
import csv
import re
import os 
import pandas as pd
import glob
from datetime import datetime, timedelta

def convert_time_to_est_ampm(time_range):
    """Convert '9:00 AM – 5:00 PM PST' to '12:00PM – 20:00PM'."""
    try:
        # Remove common time zone labels: PST, PDT, PT (case insensitive)
        time_range = re.sub(r'\s*(PST|PDT|PT)\s*', '', time_range, flags=re.IGNORECASE)

        # Normalize to use en dash
        time_range = re.sub(r'\s*[-–—]\s*', ' – ', time_range)  # handles hyphen, en dash, em dash

        start_str, end_str = [t.strip() for t in time_range.split(' – ')]

        # Parse times and convert to EST
        start_time = datetime.strptime(start_str, "%I:%M %p") + timedelta(hours=3)
        end_time = datetime.strptime(end_str, "%I:%M %p") + timedelta(hours=3)

# Manually create "HH:MMAM/PM" with 24-hour hour
        start_suffix = start_time.strftime('%p')
        end_suffix = end_time.strftime('%p')

        start_time = f"{start_time.strftime('%H:%M')}{start_suffix}"
        end_time = f"{end_time.strftime('%H:%M')}{end_suffix}"

        return f"{start_time} – {end_time}"
    
    except Exception as e:
        print(f"Time parsing error for '{time_range}': {e}")
        return time_range.strip()  # fallback

data = []

# reading html files from folder
# make sure to change the folder path to the one you want to read
folder_path = "..\Chenzi\Irvine\Irvine_Mental_Health_HTML_Files"

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.html'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the <ul> element with class "best-programs"
        ul_element = soup.find('ul', {'class': 'best-programs'})

        # Find all <li> elements within the <ul>
        li_elements = ul_element.find_all('li', {'class': 'search-result card card-v3 program-info'})

        # Loop through the found <li> elements and extract labels and values
        for li_element in li_elements:
            # Use BeautifulSoup to parse the HTML of each li_element
            li_soup = BeautifulSoup(str(li_element), 'html.parser')

            # Find the program heading 
            program_heading_element = li_soup.find('div', {'class': 'card-heading'})
            Service_org = program_heading_element.find('a', {'class': 'activity-log click-cookie'}).text 
            Service_name = program_heading_element.find('a', {'class': 'activity-log ph-flyout-click cwdc-flyout-click click-cookie'}).text

            # Combine them
            full_service = f"{Service_name} ({Service_org})"

            # Clean extra whitespace
            full_service = re.sub(r'\s+', ' ', full_service).strip()
            full_service = re.sub(r'\( ', '(', full_service)
            full_service = re.sub(r' \)', ')', full_service)
            print('Full service: ', full_service)
    
            # Extract program URL
            program_url_element = program_heading_element.find('a', {'class': 'activity-log ph-flyout-click cwdc-flyout-click click-cookie'})
            program_url = program_url_element['href'] if program_url_element else None
            Service_url = "https://www.auntbertha.com/"+program_url
            print('URL: ', Service_url)
            
            is_reviewed = False

            # Reviewed on
            Reviewd_on_element = program_heading_element.find('div', {'class': 'last-reviewed'})
            if Reviewd_on_element:  # Check if the element exists
                Reviewd_on_text = Reviewd_on_element.get_text(strip=True)  # Clean up the text
                match = re.search(r'\d{2}/\d{2}/\d{4}', Reviewd_on_text)
                Reviewd_on = match.group() if match else None
                if Reviewd_on:  # If we found a review date
                    is_reviewed = True  # Set the flag to True
                print("Reviewed on:", Reviewd_on)
            else:
                Reviewd_on = None  # Set to None if not found
                print("No review date found.")
    
            # Access class="program-tags"
            program_tags = li_soup.find('div', {'class': 'program-tags'})

            # Main Services
            main_s = []
            main_service_list = program_tags.find('ul', {'class': 'list-inline'})
            main_service_items = main_service_list.find_all('li')

            for item in main_service_items:
                main_service = item.find('a', {'class': 'loading-on-click activity-log'}).text.strip()
                main_s.append(main_service)

            print("Main Services:", main_s)

            # Try to access the "Other Services" section
            other_service_list = program_tags.find('div', {'class': 'secondary-tags'})
            if other_service_list:
                other_s = []
                other_service_items = other_service_list.find('ul', {'class': 'list-inline'}).find_all('li')

                for item in other_service_items:
                    other_service = item.find('a', {'class': 'loading-on-click activity-log'}).text.strip()
                    other_s.append(other_service)
            else:
                other_s = None

            print("Other Services:", other_s)

            # Access the "Serving" section
            serving_section = program_tags.find('div', {'class': 'attribute-tags'})

            # Initialize a list to store the serving information
            serving_ = []
    
            # Find all the <li> elements within the serving section
            serving_items = serving_section.find_all('li')

            # Iterate through the serving items and extract the text from the <a> elements
            for item in serving_items:
                link = item.find('a', {'class': 'loading-on-click activity-log'})
    
                # Check if the link was found
                if link:  # Only proceed if the link exists
                    serving_text = link.text.strip()
                    serving_.append(serving_text)

            print("Serving:", serving_)

            # accessing next-steps-module, extract phone number, location, hours 
    
            next_steps_module = li_soup.find('div', {'class': 'next-steps-module'})

            # Extract phone number
            phone_number_elements = next_steps_module.find_all('span', {'class': 'result-next-step-item'})

            # Also find all 'a' elements with href attributes containing 'tel:'
            tel_link_elements = next_steps_module.find_all('a', href=True)

            phone_number = None  # Initialize phone number as None

            # Extract phone numbers from text-based spans
            for element in phone_number_elements:
                phone_number_text = element.text.strip() if element else None
    
                # Regex to extract digits (handling separators like spaces or hyphens)
                phone_number_matches = re.findall(r'[\d-]+', phone_number_text)
    
                # Join the digits into a single phone number string if matches are found
                if phone_number_matches:
                    phone_number = ''.join(phone_number_matches)
                    break  # Stop once we've found a phone number

            # If no phone number found from text, check the 'tel:' href links
            if not phone_number:
                for element in tel_link_elements:
                    href_value = element['href']
        
                    # Check if the href contains 'tel:' and extract digits
                    if 'tel:' in href_value:
                        phone_number_matches = re.findall(r'[\d-]+', href_value)
                        if phone_number_matches:
                            phone_number = ''.join(phone_number_matches)
                            break  # Stop once we've found a phone number

            # Print the extracted phone number
            if phone_number:
                print("Phone Number:", phone_number)
            else:
                print("No phone number found.")

            # Extract location address
            location_address_element = next_steps_module.find('a', {'class': 'activity-log ph-flyout-click cwdc-flyout-click map-link with-address'})
            location_address = location_address_element.text.strip() if location_address_element else None
            location_address = re.sub(r'\s+', ' ', location_address) if location_address_element else None
            print("Location Address:", location_address)

            # Extract URL
            location_url_map = location_address_element['href'] if location_address_element else None
            print("Location url map:", location_url_map)
            
            hours_info = {
                '24_hour': False,
                'Monday': None,
                'Tuesday': None,
                'Wednesday': None,
                'Thursday': None,
                'Friday': None,
                'Saturday': None,
                'Sunday': None
            }
            
            # Extract hours based on structure
            hours_element = next_steps_module.find('div', {'class': 'office-hours-schedule see-hours-dropdown'})
            if hours_element:
                # Find all the <span> elements within the hours_element
                day_spans = hours_element.find_all('span')

                # Ensure that we have even pairs of day and corresponding hours
                if len(day_spans) % 2 == 0:
                    # Iterate through the spans in pairs (day and its corresponding hours)
                    for i in range(0, len(day_spans), 2):
                        day_span = day_spans[i]          # This is the day (e.g., "Monday:")
                        hours_span = day_spans[i + 1]    # This is the corresponding hours (e.g., "Closed" or time)

                        # Extract the day name and remove the colon
                        day = day_span.text.strip()[:-1]  # Remove the ':' at the end
                        full_day = day.strip()            # Ensure there's no extra space

                        # Check if the day is valid and exists in hours_info
                        if full_day in hours_info:
                            # Check if the corresponding hours span indicates "Closed"
                            if 'Closed' in hours_span.text:
                                hours_info[full_day] = 'Closed'
                            else:
                                # Extract hours for non-closed days
                                original_time = hours_span.text.strip()
                                converted_time = convert_time_to_est_ampm(original_time)
                                hours_info[full_day] = converted_time
                else:
                    print("Unexpected hours format. Ensure that each day has corresponding hours.")

            # Handle 24-hour information if it exists
            else:
                hours_element = next_steps_module.find('span', {'class': 'result-geo-hours'})
                if hours_element:
                    # Get 24-hour text strip
                    hours_text = hours_element.get_text(strip=True)

                    # Assuming the presence of this text means 24-hour operation
                    if '24' in hours_text or '24-hour' in hours_text.lower():
                        hours_info['24_hour'] = True
                    else:
                        print("Unable to determine if the office is 24 hours.")
                else:
                    print("Hours information not available.")

            # Output the result
            print(hours_info)
    
    
            # Loop through the found <li> elements and extract labels and values
            Extra_element = li_element.find('div', {'class': 'panel-wrapper more-info-panel'})
            elig = []
            eligibility_rules_element = Extra_element.find('div', {'class': 'eligibility-rules'})
    
            # Check if eligibility_rules_element is found
            if eligibility_rules_element:
                # Check if eligibility_rules_element contains a list (ul)
                ul_element = eligibility_rules_element.find('ul')
        
                if ul_element:
                    # If it contains a list, extract list items and store them in a flat list
                    eligibility_list = [li.text.strip() for li in ul_element.find_all('li')]
                    elig.extend(eligibility_list)  # Use extend to add elements to the list directly
                    #print("Eligibility:", eligibility_list)
                else:
                    # If it doesn't contain a list, store the text as is
                    eligibility_text = eligibility_rules_element.text.strip()
                    #print("Eligibility:", [eligibility_text])  # Wrap in a list to maintain consistency
                    elig.append(eligibility_text)

            else:
                # Skip if 'eligibility-rules' class is not found
                pass
    
            print("Eligibility:", elig)
    
            # Extract Availability
            availability_element = Extra_element.find('strong', {'data-translate': 'Availability'})
            
            if availability_element:
                availability = availability_element.find_next('div', {'class': 'col-md-10'}).text.strip()
            else:
                availability = "Not specified"

            # Extract Description
            description_element = Extra_element.find('strong', {'data-translate': 'Description'})
            description = description_element.find_next('div', {'class': 'col-md-10'}).text.strip()

            # Extract Languages
            languages_element = Extra_element.find('strong', {'data-translate': 'Languages'})
            languages = languages_element.find_next('div', {'class': 'col-md-10'}).text.strip()
            languages_data = [lang.strip() for lang in languages.split(',')]
            
            # Extract Cost
            cost_element = Extra_element.find('strong', string='Cost:')
            
            if cost_element:  # Check if cost_element exists
                cost = cost_element.find_next('div', {'class': 'col-md-10'}).text.strip()
            else:
                cost = "Not specified"  # Default value if cost_element is not found
            
            # Extract Website URLs if they exist, or set them to None
            website_element = Extra_element.find('div', {'data-translation': 'Website'})
            website_url = website_element.find_next('a', {'class': 'activity-log descriptionProgramWebsite'})['href'] if website_element else None

            # Extract Facebook and Twitter URLs if they exist, or set them to None
            facebook_element = Extra_element.find('strong', {'data-translate': 'Facebook'})
            facebook_url = facebook_element.find_next('a', {'class': 'activity-log descriptionProgramFacebook'})['href'] if facebook_element else None

            twitter_element = Extra_element.find('strong', {'data-translate': 'Twitter'})
            twitter_url = twitter_element.find_next('a', {'class': 'activity-log descriptionProgramTwitter'})['href'] if twitter_element else None

            # Extract Coverage Area
            coverage_element = Extra_element.find('strong', {'data-translate': 'Coverage Area'})
            coverage = coverage_element.find_next('div', {'class': 'col-md-10'}).text.strip()
            
            # Initialize latitude and longitude as None by default
            latitude = None
            longitude = None
            zipcode = ""

            # Find the element with class "office-hour-address"
            location_element = Extra_element.find('div', {'class': 'office-hours-address _js_address address notranslate'})

            # Check if the element exists and has the required attributes
            if location_element:
                latitude = location_element['data-latitude'] if location_element.has_attr('data-latitude') else None
                longitude = location_element['data-longitude'] if location_element.has_attr('data-longitude') else None
                print("Latitude:", latitude)
                print("Longitude:", longitude)
                
                # Extract all text within location_element
                address_text = location_element.get_text(separator=" ").strip()  # Get all text as a single string

                # Use regular expression to search for the ZIP code pattern anywhere in the text
                zip_matches = re.findall(r'(?<!\d)(\b\d{5}\b)(?!\d)', address_text)
                zipcode = zip_matches[-1] if zip_matches else ""
                print("ZIP Code:", zipcode)
                
            Google_Reviews = None

            # Change based on what the service is
            Service_Type = "Shelter"

            # Print or use the extracted values as needed
            print("Availability:", availability)
            print("Description:", description)
            print("Languages:", languages_data)
            print("Cost:", cost)
            print("Facebook URL:", facebook_url)
            print("Twitter URL:", twitter_url)
            print("Coverage Area:", coverage)

            data.append([
            full_service,
            Service_url,
            main_s,
            other_s,
            serving_,
            phone_number,
            website_url,
            location_address,
            location_url_map,
            elig,
            availability,
            description,
            languages_data,
            cost,
            is_reviewed,
            facebook_url,
            twitter_url,
            coverage,
            latitude,
            longitude,
            zipcode,
            hours_info['24_hour'],
            hours_info['Monday'],      
            hours_info['Tuesday'],     
            hours_info['Wednesday'],   
            hours_info['Thursday'],   
            hours_info['Friday'],     
            hours_info['Saturday'],  
            hours_info['Sunday'],
            Google_Reviews,
            Service_Type])
            print("***************************************************************************")

# Save to CSV file 
# Make sure to change the filename to the one you want otherwise you'll create mutliple files
csv_filename = "FindHelp_extracted_data_irvine_mental_health.csv"

with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # The header 
    header = [
        "Service_name",
        "Service_url",
        "Main_Services",
        "Other_Services",
        "Serving",
        "Phone_Number",
        "Website",
        "Location_Address",
        "Location_URL_Map",
        "Eligibility",
        "Availability",
        "Description",
        "Languages",
        "Cost",
        "Google_Review",
        "Facebook_URL",
        "Twitter_URL",
        "Coverage",
        "Latitude",
        "Longitude",
        "Zipcode",
        "24hour",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Google_Reviews",
        "Service_Type"
    ]
    csv_writer.writerow(header)
    
    csv_writer.writerows(data)

# Load ZIP codes CSV
zipcodes_df = pd.read_csv("../Misc/Zipcodes.csv")

# Choose the target place/column
target_place = "LA County"

# Get ZIP codes for that place (drop NaNs, convert to list of ints)
zipcodes_list = zipcodes_df[target_place].dropna().astype(int).tolist()

# Load the main data
df = pd.read_csv("FindHelp_extracted_data_irvine_mental_health.csv")

# Filter by ZIP codes or missing location
filtered_df = df[df['Zipcode'].isin(zipcodes_list) | (df['Location_Address'].isna() & df['Longitude'].isna() & df['Latitude'].isna())]

# Drop duplicates
filtered_df = filtered_df.drop_duplicates(subset=['Website', 'Service_name'])

filtered_df.to_csv("FindHelp_extracted_data_irvine_mental_health.csv", index=False, encoding='utf-8')

filtered_df