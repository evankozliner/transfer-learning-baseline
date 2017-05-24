if [ "$#" -eq 1 ]
then
	#touch ../../classifier/eval/tmp.txt
	#touch ../../classifier/train/tmp.txt

	#rm ../../classifier/eval/*
	#rm ../../classifier/train/*
	
	#eval "aws s3 cp s3://melonoma-classification/'$1'/eval ../../classifier/eval --recursive"
	#eval "aws s3 cp s3://melonoma-classification/'$1'/train ../../classifier/train --recursive"

	mkdir ../../tmp
	eval "aws s3 cp s3://melonoma-classification/'$1' ../../tmp --recursive"
else
	echo No bucket specified. Copy and paste a bucket from below:
	aws s3 ls melonoma-classification
fi
