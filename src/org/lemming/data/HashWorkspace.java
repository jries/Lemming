package org.lemming.data;

import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/**
 * This class is the main implementation of the Workspace abstraction. It uses a HashMap to associate a member name to an ArrayList. 
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class HashWorkspace implements Workspace {
	
	/**
	 * Creates a HashWorkspace compatible (i.e. with the same members) to h. If copyRowsToo then all elements of h are also 
	 * copied.
	 * 
	 * @param h the workspace to be copied from
	 * @param copyRowsToo copies all elements too
	 */
	public HashWorkspace(HashWorkspace h, boolean copyRowsToo) {
		for (String col : h.table.keySet())
			addNewMember(col);
		
		if (copyRowsToo)
			addAll(h);
	}
	
	/**
	 * Creates an empty workspace. There are no members.
	 * 
	 */
	public HashWorkspace() {
	}

	HashMap<String, ArrayList> table = new HashMap<String, ArrayList>();
	
	String  xVarName = "x", 
			yVarName = "x",
			zVarName = "z",
			IDVarName = "id",
			frameVarName = "frame",
			chanVarName = "channel"; 
	
	int nRows = 0;
	
	@Override
	public int getNumberOfRows() {
		return nRows;
	}
	
	@Override
	public List getMember(String members) {
		return table.get(members);
	}
	
	@Override
	public boolean hasMember(String members) {
		return table.containsKey(members);
	}

	/**
	 * Note: this operation is O(N) with the number of rows of the workspace. All 'member' of existing elements are set to null.
	 */
	@Override
	public void addNewMember(String member) {
		int N = getNumberOfRows();
		
		ArrayList l = new ArrayList();
		for (int i = 0; i<N; i++)
			l.add(null);	// NULLPOINTEREXCEPTION??
		
		table.put(member, l);
	}
	
	@Override
	public GenericLocalization getRow(int row) {
		return new GenericLocalizationI(row);
	}

	class GenericLocalizationI implements GenericLocalization {
		int rowN; 
		
		public GenericLocalizationI(int rowN) {
			this.rowN = rowN;
		}
		
		@Override
		public long getID() {
			return (long) get(IDVarName);
		}

		@Override
		public double getX() {
			return (double) get(xVarName);
		}

		@Override
		public double getY() {
			return (double) get(yVarName);

		}

		@Override
		public Object get(String member) {
			if (!has(member))
				throw new RuntimeException("Workspace has no column "+member);
			
			return table.get(member).get(rowN);
		}

		@Override
		public boolean has(String member) {
			return hasMember(member);
		}

		@Override
		public long getFrame() {
			return (long) get(frameVarName);
		}

		@Override
		public double getZ() {
			return (double) get(zVarName);
		}

		@Override
		public int getChannel() {
			return (int) get(zVarName);
		}
		
		@Override
		public void setX(double x) {
			set(xVarName, x);
		}
		
		@Override
		public void setY(double y) {
			set(yVarName, y);
		}
		
		@Override
		public void setZ(double z) {
			set(zVarName, z);
		}
		
		@Override
		public void setFrame(long frame) {
			set(frameVarName, frame);
		}
		
		@Override
		public void setChannel(int ch) {
			set(chanVarName, ch);
		}
		
		@Override
		public void setID(long ID) {
			set(IDVarName, ID);
		}

		@Override
		public void set(String member, Object o) {
			if (!has(member))
				throw new RuntimeException("Workspace has no column "+member);
			
			table.get(member).set(rowN, o);
		}
		
	}

	@Override
	public void setXname(String name) {
		xVarName = name;
	}

	@Override
	public void setYname(String name) {
		yVarName = name;
	}

	@Override
	public void setZname(String name) {
		zVarName = name;
	}

	@Override
	public void setChannelName(String name) {
		chanVarName = name;
	}

	@Override
	public void setFrameName(String name) {
		frameVarName = name;
	}

	@Override
	public void setIDname(String name) {
		IDVarName = name;
	}

	@Override
	public void addRow(GenericLocalization g) {
		for (String col : table.keySet()) {
			Object o = g.get(col);
			table.get(col).add(o);
		}
		
		nRows++;
	}

	@Override
	public GenericLocalization newRow() {
		for (List l : table.values())
			l.add(null);
		
		nRows++;
		
		return getRow(getNumberOfRows()-1);
	}

	@Override
	public void addRows(Workspace g, int from, int to) {
		for (int el = from; el<=to; el++)
			g.addRow(getRow(el));
	}

	@Override
	public void addAll(Workspace g) {
		for (GenericLocalization el : g)
			addRow(el);		
	}

	@Override
	public Iterator<GenericLocalization> iterator() {
		return new Iterator<GenericLocalization>() {
			int curElement = 0;

			public boolean hasNext() {
				return curElement < getNumberOfRows();
			}

			public GenericLocalization next() {
				return new GenericLocalizationI(curElement++);
			}

			public void remove() {
				deleteRow(curElement);
			}
		};
	}

	@Override
	public void deleteRow(int row) {
		for(List col : table.values())
			col.remove(row);
		
		nRows--;		
	}
	
	@Deprecated
	public Store<Localization> getLIFO() {
		return new Store<Localization> () {
			int lastRow = getNumberOfRows()-1; 
			
			@Override
			public boolean isEmpty() {
				return lastRow < 0;
			}
			
			@Override
			public Localization get() {
				if (isEmpty()) return null;
				
				Localization row = getRow(lastRow--);
				
				return row;
			}
			
			@Override
			public void put(Localization el) {
				GenericLocalization g = (GenericLocalization) el;
				for (String col : table.keySet()) {
					Object o = g.get(col);
					table.get(col).add(0,o);
				}
				
				nRows++;		
			}
		};
	}
	
	/**
	 * This method provides a bridge between the Workspace abstraction and the Store abstraction. 
	 * 
	 * It creates a mutable view on the workspace which allows a module working with Stores to have read/write access to the Workspace in a 
	 * first-in-first-out order using the methods provided by the Store interface. 
	 * 
	 * The put method adds the Localization to the end of the table, the get keeps track of the last row read. 
	 *  
	 * Note. If the put method is called with a GenericLocalization, then all the fields of the GenericLocalization are kept. That is, if two
	 * workspaces are connected with the FIFO interface, a put will copy the row for all members. It assumes the two workspaces are compatible, so 
	 * expect an error if they are not.
	 *  
	 * @return a class implementing the Store<Localization> interface.
	 */
	public Store<Localization> getFIFO() {
		return new Store<Localization> () {
			int lastRow = 0; 
			
			@Override
			public boolean isEmpty() {
				return lastRow >= nRows;
			}
			
			@Override
			public Localization get() {
				if (isEmpty()) return null;
				
				Localization row = getRow(lastRow++);
				
				return row;
			}
			
			@Override
			public void put(Localization el) {
				if (el instanceof GenericLocalization) {
					GenericLocalization g = (GenericLocalization) el;
					for (String col : table.keySet()) {
						Object o = g.get(col);
						table.get(col).add(o);
					}
					
					nRows++;		
				} else {
					try {
						BeanInfo b = Introspector.getBeanInfo(el.getClass());
						
						GenericLocalization g = newRow();
						for (PropertyDescriptor p : b.getPropertyDescriptors()) {
							String prop = p.getName();
							if (!g.has(prop))
								addNewMember(prop);
							g.set(prop, p.getReadMethod().invoke(el));
						}
						
					} catch (IntrospectionException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
						e.printStackTrace();
					}
				}
			}
		};
	}
	
	/**
	 * Check if an object c is compatible with the workspace. It checks whether all fields from the class are members in the workspace.
	 * 
	 * @param c the object to be tested
	 * 
	 * @return true if the object is compatible with the Workspace
	 */
	public boolean isCompatible(Object c) {
		try {
			BeanInfo b = Introspector.getBeanInfo(c.getClass());
			for (PropertyDescriptor p : b.getPropertyDescriptors()) {
				String prop = p.getName();
				if (!hasMember(prop))
					return false;
			}
		} catch (IntrospectionException e) {
			e.printStackTrace();
			
			return false;
		}
		
		return true;
	}
	
	@Override
	public String toString() {
		String out = "";
		
		// Write header
		for (String col : table.keySet())
			out += col + "\t";
		out += "\n";
		
		// Write rows
		for (int i=0; i<nRows; i++) {
			for (String col : table.keySet())
				out += table.get(col).get(i) + "\t";
			out += "\n";
		}
		
		return out;
	}

	@Override
	public String getXname() {
		return xVarName;
	}

	@Override
	public String getYname() {
		return yVarName;
	}

	@Override
	public String getZname() {
		return zVarName;
	}

	@Override
	public String getChannelName() {
		return chanVarName;
	}

	@Override
	public String getFrameName() {
		return frameVarName;
	}

	@Override
	public String getIDname() {
		return IDVarName;		
	}
	
}
