@REM Convert NED
python ConvertNED.py train/Bag Acc
python ConvertNED.py train/Bag LAcc
python ConvertNED.py train/Bag Gyr
python ConvertNED.py train/Bag Mag
python ConvertNED.py train/Hips Acc
python ConvertNED.py train/Hips LAcc
python ConvertNED.py train/Hips Gyr
python ConvertNED.py train/Hips Mag
python ConvertNED.py train/Torso Acc
python ConvertNED.py train/Torso LAcc
python ConvertNED.py train/Torso Gyr
python ConvertNED.py train/Torso Mag
python ConvertNED.py train/Hand Acc
python ConvertNED.py train/Hand LAcc
python ConvertNED.py train/Hand Gyr
python ConvertNED.py train/Hand Mag
python ConvertNED.py validation/Bag Acc
python ConvertNED.py validation/Bag LAcc
python ConvertNED.py validation/Bag Gyr
python ConvertNED.py validation/Bag Mag
python ConvertNED.py validation/Hips Acc
python ConvertNED.py validation/Hips LAcc
python ConvertNED.py validation/Hips Gyr
python ConvertNED.py validation/Hips Mag
python ConvertNED.py validation/Torso Acc
python ConvertNED.py validation/Torso LAcc
python ConvertNED.py validation/Torso Gyr
python ConvertNED.py validation/Torso Mag
python ConvertNED.py validation/Hand Acc
python ConvertNED.py validation/Hand LAcc
python ConvertNED.py validation/Hand Gyr
python ConvertNED.py validation/Hand Mag
python ConvertNED.py test Acc
python ConvertNED.py test LAcc
python ConvertNED.py test Gyr
python ConvertNED.py test Mag
@REM numpy generate
python numpy_generate.py train Bag
python numpy_generate.py train Hips
python numpy_generate.py train Torso
python numpy_generate.py train Hand
python numpy_generate.py validation Bag
python numpy_generate.py validation Hips
python numpy_generate.py validation Torso
python numpy_generate.py validation Hand
python numpy_generate.py test a
@REM preprocess
python preprocess.py train
python preprocess.py validation
python preprocess.py test
@REM model create
python model_create.py
