if [ "$#" -lt 1 ]
then
	echo "Missing Arguments!"
	echo "Usage: ./s3_uploader.sh 'notes about run for notes.txt' 'number of runs'"
	exit 1
fi

date=$(date)
upload_path="../../classifier"

s3_path="aws s3 cp $upload_path s3://melonoma-classification/"
#echo "$s3_path$2-runs-'$date'/ --include '*' --exclude '*.bin' --recursive"
eval "$s3_path$2-'$date'/ --include '*' --exclude '*.bin' --recursive"

upload_path="notes.txt"
s3_path="aws s3 cp $upload_path s3://melonoma-classification/"
echo $1 >> $upload_path
eval "$s3_path$2-'$date'/"
rm notes.txt

upload_path="../conversation_helper/conversion_config.json"
s3_path="aws s3 cp $upload_path s3://melonoma-classification/"
eval "$s3_path$2-'$date'/"

