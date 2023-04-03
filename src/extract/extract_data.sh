#!/bin/sh

ACCESS_KEY="474b7349-e7e9-4bcc-988d-4d9d4d16d069"

# Obtain a new token
response=$(curl -s -X GET -H "AccessKey: $ACCESS_KEY" "https://www.ura.gov.sg/uraDataService/insertNewToken.action")
token=$(echo "$response" | sed -n 's/.*"Result":"\([^"]*\).*/\1/p')

#Create temp folder
mkdir -p data/raw_data

# Loop through batch parameter from 1 to 4
for i in {1..4}; do
    # Call the URA API with the obtained token and current batch parameter
    response=$(curl -s -X GET \
        -H "AccessKey: $ACCESS_KEY" \
        -H "Token: $token" \
        "https://www.ura.gov.sg/uraDataService/invokeUraDS?service=PMI_Resi_Transaction&batch=$i")
    # Save the response to a file named transactions.txt
    echo "$response" > data/raw_data/transactions_$i.txt
    echo "Saved to data/raw_data/transactions_$i.txt"
done

echo "Extraction done"
