You can run specify the part when running the code by passing the argument --part. The sentence for which the attentnion maps are generated can also be passed by the argument --sentence but it is optional. Here is an example:

 python main.py --part 1 --sentence "And third, the Congress should, this month, enact measures to increase domestic energy production and energy conservation in order to reduce dependence on foreign oil."

--part should be either 1 or 2. You can get the improvements for the third part by doing the following:
In the forward method of the TransformerClassifier class, and in the attention method of the Head class, comment and uncomment lines based on the instruction.

Please make the plots/part1 and plots/part2 directories for the plots to be generated.