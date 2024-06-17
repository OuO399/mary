import javalang

java_code = '''public class test{
public void addMessageContext(MessageContext mc) throws AxisFault {
    mc.setServiceContext(sc);
    if (mc.getMessageID() == null) {
        setMessageID(mc);
    }
    axisOp.registerOperationContext(mc, oc);
}
}

'''
tree = javalang.parse.parse(java_code)
alltokens=[]
for path, node in tree:
    if path==[] or path==():
        continue
    # print(type(node))
    alltokens.append(node.__class__.__name__)
print(type(str(alltokens)))