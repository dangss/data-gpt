# Update version for image in deployment.yml then this command affect
# Learn: https://learn.microsoft.com/en-us/azure/container-instances/container-instances-quickstart
az container create --resource-group DI_GPT_Research_US --name digpt -f deployment.yml
#    --image microsoft/public_datasets \
#    --location eastus \
#    --ports 80 \
#    --ip-address Public \
#    --azure-file-volume-account-name digptstorage \
#    --azure-file-volume-account-key K+qWugy412SH5UF7H5iDktwNjwn8L3gOriRREOUh8zGErONqa6No7UAfsSY+TPKgR49w461k9OoG+ASt6609mQ== \
#    --azure-file-volume-share-name digptdb \
#    --azure-file-volume-mount-path /aci/logs/

# run command
# Check log: az container logs --resource-group DI_GPT_Research_US --name digpt
# tail log: az container attach --resource-group DI_GPT_Research_US --name digpt
