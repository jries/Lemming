package org.lemming.inputs;

import org.lemming.data.GenericLocalization;
import org.lemming.data.Localization;
import org.lemming.data.Workspace;
import org.lemming.data.XYFLocalization;
import org.lemming.interfaces.GenericWorkspacePlugin;
import org.lemming.interfaces.IncompatibleWorkspaceException;

/**
 * 
 * 
 * @author Thomas Pengo, Joran Deschamps
 *
 */
public class WorkspaceDriftLocalizer extends GenericWorkspacePlugin implements Runnable {

	DriftXYFLocalizer r;
	
	public WorkspaceDriftLocalizer(int N) {
		r = new DriftXYFLocalizer(N);
	}

	@Override
	public void setInput(Workspace in) throws IncompatibleWorkspaceException {
	}

	Workspace out;
	
	@Override
	public void setOutput(Workspace out) {
		this.out = out;
		
		if(!out.hasMember(out.getIDname())) out.addNewMember(out.getIDname());
		if(!out.hasMember(out.getXname())) out.addNewMember(out.getXname());
		if(!out.hasMember(out.getYname())) out.addNewMember(out.getYname());
		if(!out.hasMember(out.getFrameName())) out.addNewMember(out.getFrameName());
	}

	@Override
	public String[] getRequiredMembers() {
		return NO_REQUIREMENTS;
	}

	@Override
	public void setRequiredMembers(int flags, String[] requiredMembers) {
	}

	@Override
	public void setRequiredMembers(int flags) {
	}

	@Override
	public void run() {
		while(r.hasMoreOutputs()) {
			XYFLocalization l = r.newOutput();
			
			GenericLocalization g = out.newRow();
			g.setX(l.getX());
			g.setY(l.getY());
			g.setID(l.getID());
			g.setFrame(l.getFrame());
		}
	}

}
