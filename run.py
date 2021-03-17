from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from argparse import ArgumentParser as Argp
import torch, json, re


def load_json( path ) :
    with open( path, 'r', encoding = 'UTF-8' ) as f :
        return json.load( f )


def reg_deleter( sentence, regex, after ) :
    reg = re.compile( regex, re.I )
    while reg.search( sentence ) != None :
        tool = reg.search( sentence )
        sentence = sentence.replace( tool.group( ), after )
    return sentence


def answer( tokenizer, model, device, text, question ) :
    model.to( device )
    encoding = tokenizer( question, text, return_tensors = "pt" ).to( device )
    input_ids = encoding[ "input_ids" ].to( device )
    outputs = model( **encoding )
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens( input_ids[ 0 ].tolist( ) )
    answer_tokens = all_tokens[ torch.argmax( start_logits ) :torch.argmax( end_logits ) + 1 ]
    answer = tokenizer.decode( tokenizer.convert_tokens_to_ids( answer_tokens ) )
    return answer


def write_json( res, path ) :
    with open( path, 'w', encoding = 'UTF-8' ) as f :
        json.dump( res, f, indent = 4 )
        f.close( )


def main( ) :
    arg = Argp( )
    arg.add_argument( '-d', '--device', default = 'cpu', help = 'Device to use : cuda or cpu (default : cpu)' )
    arg.add_argument( '-p', '--prepro', default = 'False', help = 'Whether to delete parenthesis (default : False)' )
    args = arg.parse_args( )
    device = args.device
    prepro = eval(args.prepro)

    models = open( r'./model.txt', 'r' ).read( ).splitlines( )
    input_data = load_json( r'./input/input.json' )
    text = input_data[ 'text' ]
    if prepro == True :
        text = reg_deleter( text, '\([^\(\)]*\)', '' )
    questions = input_data[ 'question' ]

    result = dict( )
    result[ 'text' ] = text
    result[ 'QA' ] = list( )

    for n1, m in enumerate( models ) :
        print( f"{m}" )
        name = m.split( "/" )[ 1 ].split( "-" )[ 0 ]
        tokenizer = AutoTokenizer.from_pretrained( m )
        model = AutoModelForQuestionAnswering.from_pretrained( m )
        for n2, q in enumerate( questions ) :
            ans = answer( tokenizer, model, device, text, q )
            if n1 == 0 :
                qa = { "question" : q, "answer" : { name : ans } }
                result[ 'QA' ].append( qa )
            else :
                result[ 'QA' ][ n2 ][ "answer" ][ name ] = ans
    write_json( result, r'./output/output.json' )


if __name__ == "__main__" :
    main( )
