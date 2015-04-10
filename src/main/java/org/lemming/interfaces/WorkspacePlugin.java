package org.lemming.interfaces;


/**
 * The WorkspacePlugin interface provides a plugin interface for modules to be used in a pipeline using a Workspace as input and output. 
 * It also contains the constants needed to specify the required fields within the standard fields of the workspace (X,Y,Z,frame,channel and ID).
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */

public interface WorkspacePlugin {
	/**
	 * 
	 */
	public static final int NEEDS_X = 1;
	/**
	 * 
	 */
	public static final int NEEDS_Y = 2;
	/**
	 * 
	 */
	public static final int NEEDS_Z = 4;
	/**
	 * 
	 */
	public static final int NEEDS_FRAME = 8;
	/**
	 * 
	 */
	public static final int NEEDS_CHAN = 16;
	/**
	 * 
	 */
	public static final int NEEDS_ID = 32;
	/**
	 * 
	 */
	public static final String[] NO_REQUIREMENTS = new String[] {};

	/**
	 * Checks requirements and if the Workspace is compatible, then sets the input accordingly.
	 * 
	 * @param in the input Workspace
	 * @throws IncompatibleWorkspaceException if not compatible
	 */
	public void setInput(Workspace in) throws IncompatibleWorkspaceException;
	
	/**
	 * Sets output to workspace out and creates appropriate members if necessary
	 * 
	 * @param out - output
	 */
	public void setOutput(Workspace out);

	/**
	 * @return Returns an array of Strings
	 */
	public abstract String[] getRequiredMembers();

	/**
	 * Sets the requirements on the standard members and any additional requirements.
	 * 
	 * @param flags - Flags
	 * @param requiredMembers and array of Strings with the names of the additional required members.
	 */
	public abstract void setRequiredMembers(int flags, String[] requiredMembers);

	/**
	 * Sets the requirements on the standard members.
	 * 
	 * @param flags - Flags
	 */
	public abstract void setRequiredMembers(int flags);	
}

