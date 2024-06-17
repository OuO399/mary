public class test{
    public void addMessageContext(MessageContext mc) throws AxisFault {
        mc.setServiceContext(sc);
        if (mc.getMessageID() == null) {
            setMessageID(mc);
        }
        axisOp.registerOperationContext(mc, oc);
    }
}
