from donut import DonutModel

donutmodel = DonutModel.from_pretrained('result/bscard3')
donut_decoder = donutmodel.decoder


tokenizer = donut_decoder.tokenizer

