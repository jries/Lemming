package org.lemming.interfaces;


/**
 * A default implementation of a workspace plugin. It has an input and an output and checks the input for compatibility based on the requirements.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class GenericWorkspacePlugin implements WorkspacePlugin {
	private String[] requiredMembers = new String[] {};
	
	private int flags = 0;
	
	@Override
	public void setRequiredMembers(int flags) {
		setRequiredMembers(flags, new String[] {});
	}
	
	@Override
	public void setRequiredMembers(int flags, String[] requiredMembers) {
		this.requiredMembers = requiredMembers;
		this.flags = flags;
	}

	@Override
	public String[] getRequiredMembers() {
		return requiredMembers;
	}

	/**
	 * input 
	 */
	public Workspace in;
	/**
	 * output
	 */
	public Workspace out;
	
	@Override
	public void setInput(Workspace in) throws IncompatibleWorkspaceException {
		for (String s : requiredMembers) {
			if (!in.hasMember(s))
				throw new IncompatibleWorkspaceException(s);
		}
		
		if ( (flags & NEEDS_X) > 0 && !in.hasMember(in.getXname()) ) throw new IncompatibleWorkspaceException("<X>");
		if ( (flags & NEEDS_Y) > 0 && !in.hasMember(in.getYname()) ) throw new IncompatibleWorkspaceException("<Y>");
		if ( (flags & NEEDS_Z) > 0 && !in.hasMember(in.getZname()) ) throw new IncompatibleWorkspaceException("<Z>");
		if ( (flags & NEEDS_FRAME) > 0 && !in.hasMember(in.getFrameName()) ) throw new IncompatibleWorkspaceException("<FRAME>");
		if ( (flags & NEEDS_CHAN) > 0 && !in.hasMember(in.getChannelName()) ) throw new IncompatibleWorkspaceException("<CHAN>");
		if ( (flags & NEEDS_ID) > 0 && !in.hasMember(in.getIDname()) ) throw new IncompatibleWorkspaceException("<ID>");
		
		this.in = in;
	}

	@Override
	public void setOutput(Workspace out) {
		this.out = out;
	}

}
