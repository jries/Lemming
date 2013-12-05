package org.lemming.data;

import java.util.List;

/**
 * Represents a table of generic localizations. It can be accessed with an iterator.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Workspace extends Iterable<GenericLocalization> {

	/**
	 * Return the entire column as present in the workspace at the time the method is called.
	 * 
	 * @param key
	 * @return
	 */
	public List getMember(String members);
	
	/**
	 * Checks if the specified member is present in the workspace.
	 * 
	 * @param key
	 * @return
	 */
	public boolean hasMember(String members);
	
	/**
	 * Creates a new member in the workspace, with empty members in order to keep all members
	 * of the same length (table rectangular).
	 * 
	 * @param key
	 * @return
	 */
	public void addNewMember(String members);
	
	/**
	 * Return a particular row in the table.  
	 * 
	 * @param row
	 * @return
	 */
	public GenericLocalization getRow(int row);
	
	/**
	 * Add a new row to the table. If the localization does not have a particular member, 
	 * a default value (null) will be inserted into the workspace. If the workspace does not have
	 * a particular member present in g, that member will be ignored.
	 * 
	 * @param g
	 * @return
	 */
	public void addRow(GenericLocalization g);
	
	/**
	 * Removes the specified row.
	 * 
	 * @return
	 */
	public void deleteRow(int row);
	
	/**
	 * Copy all rows from "from" to "to" into the current workspace.
	 * 
	 * @param g
	 * @param from 
	 * @param to
	 * @return
	 */
	public void addRows(Workspace g, int from, int to);
	
	/**
	 * Copy all rows from g to the current workspace.
	 * 
	 * @param g
	 * @return
	 */
	public void addAll(Workspace g);
	
	public void setXname(String name);
	public void setYname(String name);
	public void setZname(String name);
	public void setChannelName(String name);
	public void setFrameName(String name);
	public void setIDname(String name);
	
	public String getXname();
	public String getYname();
	public String getZname();
	public String getChannelName();
	public String getFrameName();
	public String getIDname();
	
	public GenericLocalization newRow();

	/**
	 * Returns the number of rows in the table.
	 * 
	 * @return
	 */
	int getNumberOfRows();
	
}
