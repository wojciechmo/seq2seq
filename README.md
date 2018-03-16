# Sequence to Sequence

PyTroch implementation of Sequence to Sequence model for machine translation. 

**Attention mechanism** allow decoder decide on which input words to focus during generation of each output word. 

<img src="https://s13.postimg.org/si1crybrr/no_hay_rey_en_polonia-_there_is_no_king_in_there.png" width="220"> 

In Sequence to Sequence model two recurrent neural networks work together to transform one sequence to another. 

**Encoder** reads an input sequence and outputs single context vector and hidden states from every timestep. 

<img src="https://s17.postimg.org/tue57sq33/image.png" width="400">

**Decoder** recieves context vector, and with the help of encoder hidden states produces an output sequence.

<img src="https://s13.postimg.org/gjzsdui8n/image.png" width="600">

Network generally learns to translate unseen sentences from Spanish to English. Sometimes translations are perfect, sometimes mistakes are minor and sometimes output have little common with input. All things considered, it suprassed my expectations. Trainig was done around 100 000 pairs of spanish-english sentences. Spanish vocabulary size was around 16 000 and English around 8 000 words. Embeddings were attached to inputs both in encoder and decoder. 

**Perfect translations:**<br />
Conoci al rey. - I met the king.<br />
El deporte es salud. - Sport is health.<br />
Ellos tienen mucho dinero. - They have a lot money.<br />
Francia es un pais muy interesante. - France is a very interesting country.<br />
Hoy me levanto temprano en la manana. - I got up early in the morning.<br />
La vida es hermosa. - Life is beautiful.<br />
Me gusta leer libros. - I like to read books.<br />
Mucha gente vive en japon. - A lot of people live in japan.<br />
No me gusta leer libros. - I don't like to read books.<br />
Tu perro y tu gato viven bien. - Your dog and your cat live well.<br />

**Minor mistakes:**<br />
El baloncesto es popular en america - Let's is popular in america.<br />
El es sabio pero tambien ingenuo. - He is an but but naive.<br />
En invierno esta nevando y lloviendo. - Winter is in in it is raining.<br />
Espana ataco a francia. - We were attacked to france.<br />
Ganamos mucho dinero ayer. - We drank a lot of money yesterday.<br />
La historia le gusta repetir. - The story likes to.<br />
La vida es bella pero tambien pesada. - Life is but also as heavy.<br />
Me gusta correr mucho por la manana. - I like to run a lot of the.<br />
Me gusta leer libros y cantar. - I like reading books and sing.<br />
Rusia es el pais mas grande de europa y del mundo. - Russia is the largest country in europe and in the.<br />
Solia ser mejor en el pasado. - It used to be better at the past.<br />

**Serious mistakes:**<br />
Ayer monte un carro. - I was a car car.<br />
Cuando visitamos a los padres. - When was the parents parents.<br />
En japon mucha gente vive. - She is many people.<br />
En la ciudad puedes andar en bicicleta y en automovil. - In town can walk a and and.<br />
Estuve en el cine ayer. - I was the the the yesterday yesterday.<br />

More attentions examples:

No lo hago por dinero. - I don't do it for money.<br />
Nunca he estado en china. - I've never been to in china.<br />
En america hay gasolina barata. - There are in the only in there are.<br />

<img src="https://s13.postimg.org/nn7ntrdh3/canvas.png" height="277">


